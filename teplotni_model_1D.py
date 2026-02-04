import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# ==============================================================================
# ==============================================================================
#                       1. UŽIVATELSKÉ NASTAVENÍ (KONFIGURACE)
# ==============================================================================
# ==============================================================================
# Zde nastavujete hlavní "přepínače" simulace. 

# --- A. NASTAVENÍ CHLAZENÍ (PRO FÁZI ZÁBĚRU) ---
# Jakým způsobem má model počítat odvod tepla (součinitel 'h')?
# MOŽNOSTI:
#   "STATIC"     = Použije fixní hodnotu h.
#   "ANALYTIC"   = (DOPORUČENO) Dynamicky počítá h podle otáček a viskozity.
#   "FLOW_LIMIT" = Počítá maximální možné chlazení dle kapacity průtoku.
#   "NO_COOLING" = Vypne chlazení (h = 0).

CHLAZENI_TYP = "ANALYTIC"   # <--- ZDE ZMĚŇTE TYP CHLAZENÍ

# Máme započítat, že drážky snižují stykovou plochu?
INCLUDE_AREA_REDUCTION = False   # <--- ZDE ZMĚŇTE VOLBU

# --- B. ODBĚR VÝKONU NÁSTAVBOU ---
# Výkon, který odebírá hydraulika přímo z motoru.
P_auxiliary_load_kW = 100.0  # [kW]

# --- C. TEPLOTY OLEJE ---
# Teplota oleje na vstupu (určuje chlazení i viskozitu).
T_olej_inlet = 70.0   # [°C]

# --- D. NASTAVENÍ ROZJEZDU DO KOPCE (HILL START) ---
# Tato sekce umožňuje simulovat fázi "předzáběru", kdy auto stojí v kopci.

ENABLE_HILL_START = False   # Zapnout simulaci kopce? (True/False)

# Parametry pro HILL START (Použijí se jen když ENABLE_HILL_START = True)
t_hold       = 1.0    # [s]   Doba držení auta na spojce (předzáběr)
n_motor_hold = 800.0  # [rpm] Otáčky motoru při stání v kopci.

# --- E. PRŮBĚH OTÁČEK (SAMOTNÝ ROZJEZD - LAUNCH) ---
# Tvarování průběhu otáček (Shape Factor)
# Exponent = 1.0 (Lineární), > 1.0 (Konkávní - Jízda na spojce)
RPM_SHAPE_FACTOR = 1.0 

# 1. OTÁČKY MOTORU (Absolutní)
n_motor_start = 1200.0  # [rpm] Start rozjezdu
n_motor_end   = 1200.0  # [rpm] Konec prokluzu
n_motor_idle  = 1200.0   # [rpm] Otáčky motoru v PAUZE (Důležité pro Drag Torque!)

# 2. OTÁČKY PROKLUZU (Relativní rozdíl)
n_slip_start  = 1200.0  # [rpm] Počáteční prokluz
n_slip_end    = 0.0     # [rpm] Konec prokluzu

# --- F. ČASOVÁNÍ CYKLU ---
n_cyklu = 8           # Počet opakování
t_zab   = 1.74        # [s] Doba prokluzu
t_pauza = 30.0         # [s] Doba chladnutí (a generování drag torque)

# --- G. CHOVÁNÍ V PAUZE (OPEN CLUTCH & DRAG TORQUE) ---

# 1. Chlazení rozepnuté spojky (Větší plocha)
ENABLE_OPEN_CLUTCH_COOLING = True  # Povolit "Flow Limit" na celou plochu v pauze
ratio_pause = 1.0                  # 100% plochy se chladí

# 2. NOVÉ: Vlečný moment (Drag Torque)
# Počítá hydraulický odpor oleje mezi rozpojenými lamelami.
# Způsobuje ohřev i v pauze.
ENABLE_DRAG_TORQUE = True     # <--- ZAPNUTÍ VÝPOČTU ODPORU
h_gap_mm = 0.2               # [mm] Vůle mezi lamelou a diskem (typicky 0.1 - 0.3 mm)
                              # Menší mezera = Větší odpor (nepřímo úměrné).

# Ošetření logiky: Pokud je Hill Start vypnutý, holding time je 0
if not ENABLE_HILL_START:
    t_hold = 0.0
# ==============================================================================
# ==============================================================================
#                       2. DEFINICE FYZIKÁLNÍCH FUNKCÍ
# ==============================================================================
# ==============================================================================

def get_steel_props(T_celsius):
    """
    Vypočítá materiálové vlastnosti uhlíkové oceli v závislosti na teplotě.
    Ocel mění své vlastnosti s teplotou, což model zohledňuje (nelinearita).
    """
    # Oříznutí teploty pro bezpečnost (aby vzorce neulétly v extrému)
    T = np.clip(T_celsius, 20.0, 1000.0)
    
    # 1. Tepelná vodivost k [W/m.K] - Schopnost vést teplo.
    # U oceli klesá s teplotou (za studena vede lépe).
    k = 54.0 - 0.028 * T
    
    # 2. Měrná tepelná kapacita cp [J/kg.K] - Schopnost akumulovat teplo.
    # U oceli roste s teplotou.
    c_p = 450.0 + 0.28 * T
    
    # 3. Hustota rho [kg/m3] - Hmotnost objemu. Považujeme za konstantu.
    rho = 7850.0
    
    return k, c_p, rho

def get_cooling_analytical(rpm, T_viscosity_input, geometry):
    """
    POKROČILÝ VÝPOČET SOUČINITELE PŘESTUPU TEPLA 'h' (neboli alpha)
    --------------------------------------------------------------
    
    Vstupy:
      rpm               : Aktuální otáčky koše spojky (motoru)
      T_viscosity_input : Teplota použitá PRO URČENÍ VISKOZITY.
      geometry          : Slovník s rozměry (poloměry, hydraulický průměr)
    """
    # Ošetření nulových otáček (vždy je tam alespoň malá konvekce)
    if rpm < 10: return 50.0 
    
    # --- A. FYZIKÁLNÍ VLASTNOSTI OLEJE ---
    rho = 850.0       # Hustota [kg/m3]
    lam_oil = 0.14    # Tepelná vodivost oleje (lambda) [W/m.K]
    c_oil = 2000.0    # Tepelná kapacita oleje (cp) [J/kg.K]
    
    # Kinematická viskozita (nu) [m2/s] interpolovaná dle teploty
    nu = np.interp(T_viscosity_input, [40, 100], [30e-6, 7e-6]) 
    
    # --- B. GEOMETRIE ---
    r_in = geometry['r_in']
    r_out = geometry['r_out']
    Dh = geometry['Dh'] # Hydraulický průměr (d_h)
    
    # --- C. VÝPOČET DLE HYDRODYNAMIKY ---
    
    # 1. Úhlová rychlost [rad/s]
    omega = rpm * (2 * np.pi / 60)
    
    # 2. Rychlost toku v drážce (v_r)
    # Rovnice: v = omega * sqrt(r2^2 - r1^2)
    v_oil = omega * np.sqrt(r_out**2 - r_in**2)
    
    # 3. Reynoldsovo číslo (Re)
    Re = (v_oil * Dh) / nu
    
    # 4. Prandtlovo číslo (Pr)
    Pr = (rho * c_oil * nu) / lam_oil
    
    # 5. Nusseltovo číslo (Nu)
    # Dittus-Boelterova korelace pro turbulentní tok
    Nu = 0.023 * (Re**0.8) * (Pr**0.3)
    
    # 6. Součinitel přestupu tepla h (neboli alpha)
    h_pipe = (Nu / Dh) * lam_oil
    
    # KOREKCE NA REALITU SPOJKY (Enhancement Factor)
    # Zohledňuje drsnost, vstupní efekty a míchání v drážkách
    enhancement_factor = 1.0 
    
    h = h_pipe * enhancement_factor
    
    return h

def get_drag_torque_analytical(rpm_slip, T_viscosity_input, geometry, h_gap_mm):
    """
    NOVÁ FUNKCE: VÝPOČET VLEČNÉHO MOMENTU (DRAG TORQUE)
    ---------------------------------------------------
    Počítá moment, který přenáší viskozita oleje mezi rozpojenými lamelami.
    Vychází z analytického řešení Navier-Stokesových rovnic pro laminární smykové proudění 
    mezi dvěma rotujícími mezikružími.
    
    Vzorec: M = (pi * mu * omega * (R_out^4 - R_in^4)) / (2 * h_gap)
    
    Vstupy:
      rpm_slip          : Rozdíl otáček mezi motorem a převodovkou (prokluz)
      T_viscosity_input : Teplota pro určení viskozity (zde T_olej_inlet)
      geometry          : Slovník s poloměry
      h_gap_mm          : Vůle mezi lamelami v mm
    """
    # 1. Viskozita oleje (stejná interpolace jako u chlazení)
    rho = 850.0
    # Interpolace kinematické viskozity (nu)
    nu = np.interp(T_viscosity_input, [40, 100], [30e-6, 7e-6]) 
    # Výpočet dynamické viskozity (mu) [Pa.s] = nu * rho
    mu = nu * rho 
    
    # 2. Geometrie a převody jednotek
    r_out = geometry['r_out']
    r_in = geometry['r_in']
    h_gap = h_gap_mm / 1000.0 # Převod z mm na metry [m]
    
    # 3. Úhlová rychlost prokluzu [rad/s]
    omega_slip = rpm_slip * (2 * np.pi / 60)
    
    # 4. Výpočet momentu
    # Tento vzorec integruje smykové napětí tau = mu * (v/h) * r po ploše.
    # Protože v = omega * r, je tau závislé na r.
    # Výsledkem je závislost na čtvrté mocnině poloměru (velký vliv průměru!).
    
    term_geo = (r_out**4 - r_in**4)
    M_drag = (np.pi * mu * omega_slip * term_geo) / (2 * h_gap)
    
    return M_drag

def load_engine_map(filename='motor_data.xlsx'):
    """ Načte charakteristiku motoru z Excelu a vytvoří interpolační funkci. """
    try:
        df = pd.read_excel(filename)
        rpm_data = df['RPM'].values
        torque_data = df['Torque'].values
        print(f"ÚSPĚCH: Načten soubor '{filename}'.")
    except FileNotFoundError:
        print(f"CHYBA: Soubor '{filename}' nenalezen! (Používám demo data pro ukázku)")
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
    except KeyError:
        print("CHYBA: Excel musí mít sloupce 'RPM' a 'Torque'.")
        raise
    # Vytvoří "spojitou čáru" z bodů (interpolace)
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data

# ==============================================================================
# ==============================================================================
#                       3. INICIALIZACE A GEOMETRIE
# ==============================================================================
# ==============================================================================

# Načtení mapy motoru
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()

# --- MATERIÁLOVÉ KONSTANTY ---
# Referenční hodnoty pro výpočet koeficientu Beta (rozdělení tepla)
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2 # Třecí obložení (papír/karbon)

# Koeficient Beta - Kolik % tepla jde do oceli?
# Vypočteno z poměru tepelných jímavostí (b = odmocnina(k*rho*c))
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

# Vlastnosti oleje (pro statické výpočty)
c_oil = 2000.0       # Kapacita oleje
lambda_oil = 0.14    # Vodivost

# --- GEOMETRIE SPOJKY ---
r_out = 0.124        # [m] Vnější poloměr
r_in = 0.0875        # [m] Vnitřní poloměr
tloustka_oceli = 0.004 # [m]
T_skrin = T_olej_inlet # Počáteční teplota

# --- GEOMETRIE DRÁŽEK (WAFFLE) ---
sirka_drazky = 0.0015  # [m]
hloubka_drazky = 0.0002 # [m]
roztec_drazek = 0.009  # [m]

# Výpočet hydraulického průměru (Dh) - "Efektivní průměr trubky"
S_tok = sirka_drazky * hloubka_drazky
O_tok = 2 * (sirka_drazky + hloubka_drazky)
Dh = 4 * S_tok / O_tok

# Slovník pro předávání do funkcí
geometry_dict = {'r_in': r_in, 'r_out': r_out, 'Dh': Dh}

# --- PLOCHY A REDUKCE ---
S_celkova_mezikruzi = np.pi * (r_out**2 - r_in**2)
ratio_groove = 0.06  # Kolik % plochy zabírají drážky
S_cooling = S_celkova_mezikruzi * ratio_groove          # Plocha kudy teče olej
S_contact = S_celkova_mezikruzi * (1 - ratio_groove)    # Skutečná plocha kam jde teplo (obložení-ocel)

# Aplikace volby uživatele (INCLUDE_AREA_REDUCTION)
if INCLUDE_AREA_REDUCTION:
    S_calc_power = S_contact 
    area_note = "Redukovaná (Odečteny drážky)"
else:
    S_calc_power = S_celkova_mezikruzi 
    area_note = "Celková (Ignorovány drážky)"

n_pairs = 14  # Počet třecích ploch

# --- PŘÍPRAVA HODNOT PRO ALTERNATIVNÍ REŽIMY CHLAZENÍ (PRO VÝPIS) ---
# Zde si předpřipravíme hodnoty limitů, abychom je mohli použít v logice Open Clutch.

q_total_lmin = 6.0 # L/min (Celkový průtok do spojky)
mdot_per_surface = (q_total_lmin / 60 / 1000 * 850) / n_pairs 

# 1. Flow Limit pro DRÁŽKY (Standardní režim)
h_flow_limit_grooves = (mdot_per_surface * c_oil) / S_cooling

# 2. Flow Limit pro PAUZU (Rozepnuto - NOVÉ)
# Pokud v pauze chladíme větší plochu (ratio_pause), h se musí přepočítat.
# Energie odnesená olejem je stejná (m*c*dT), ale rozpočítá se na větší plochu.
S_cooling_pause = S_celkova_mezikruzi * ratio_pause
h_flow_limit_pause = (mdot_per_surface * c_oil) / S_cooling_pause

# 3. Statické h (fixní - laminární)
Nu_stat = 6.05
h_static_val = (Nu_stat * lambda_oil) / Dh

# 4. Analytické h (Demo hodnota pro startovní otáčky)
# Použijeme funkci, kterou máme, abychom uživateli ukázali, kolik to bude na začátku
h_analytic_demo_val = get_cooling_analytical(n_motor_start, T_olej_inlet, geometry_dict)

# Výpis info do konzole - PŘEHLEDNÁ TABULKA
print("-" * 60)
print(f"INFO: NASTAVENÍ SIMULACE")
print(f"  * Režim chlazení (ZVOLENÝ): {CHLAZENI_TYP}")
print(f"  * Tvar otáček (Exponent):   {RPM_SHAPE_FACTOR} (1=Lin, >1=Konkávní, <1=Konvexní)")
print(f"  * Chlazení v pauze (Open):  {'FLOW LIMIT (Aktivní)' if (ENABLE_OPEN_CLUTCH_COOLING and CHLAZENI_TYP=='ANALYTIC') else 'Standardní'}")
if ENABLE_OPEN_CLUTCH_COOLING and CHLAZENI_TYP == "ANALYTIC":
    print(f"    -> Plocha v pauze:        {ratio_pause*100:.0f} % (Lamely odskočeny)")
    print(f"    -> h v pauze (limit):     {h_flow_limit_pause:.1f} W/m2K")

# --- NOVÝ VÝPIS PRO VLEČNÝ MOMENT ---
print(f"  * Vlečný moment (Drag):     {'AKTIVNÍ' if ENABLE_DRAG_TORQUE else 'VYPNUTO'}")
if ENABLE_DRAG_TORQUE:
    print(f"    -> Vůle mezi lamelami:    {h_gap_mm} mm")
    print(f"    -> Otáčky v pauze:        {n_motor_idle} rpm")

print(f"  * Teplota přívodu oleje:    {T_olej_inlet} °C")
print(f"  * Výkon nástavby (odběr):   {P_auxiliary_load_kW} kW")
print(f"  * Rozjezd do kopce:         {'AKTIVNÍ' if ENABLE_HILL_START else 'NEAKTIVNÍ'}")
if ENABLE_HILL_START:
    print(f"    -> Doba držení (Hold):    {t_hold} s")
    print(f"    -> Otáčky držení:         {n_motor_hold} rpm")
print("-" * 60)
print("PŘEHLED MOŽNÝCH HODNOT CHLAZENÍ (PRO INFO A POROVNÁNÍ):")
print(f"  1. STATIC (Laminární):     {h_static_val:.1f} W/m2K")
print(f"  2. FLOW LIMIT (Drážky):    {h_flow_limit_grooves:.1f} W/m2K (při {q_total_lmin} L/min)")
print(f"  3. ANALYTIC (při {n_motor_start:.0f} rpm): {h_analytic_demo_val:.1f} W/m2K (při viskozitě pro {T_olej_inlet}°C)")
print(f"  4. NO COOLING:             0.0 W/m2K")
print("-" * 60)

# ==============================================================================
# ==============================================================================
#                       4. HLAVNÍ SIMULAČNÍ SMYČKA (SOLVER)
# ==============================================================================
# ==============================================================================

# Celkový čas cyklu se prodlouží o dobu držení v kopci (t_hold)
t_cyklus = t_hold + t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Nastavení sítě pro MKP/FDM (Metoda konečných diferencí)
# Dělíme tloušťku oceli na N malých vrstev.
L = tloustka_oceli / 2 # Počítáme jen polovinu (symetrie)
N = 50                 # Počet uzlů
dx = L / (N - 1)       # Vzdálenost mezi uzly [m]

# Časový krok (dt) - Musí být dost malý pro stabilitu výpočtu
k_c, c_c, rho_c = get_steel_props(20.0)
dt = 0.9 * (0.5 * dx**2 / (k_c / (rho_c * c_c))) # Courantovo kritérium

# Inicializace teplotního pole (ve všech bodech je na začátku T_skrin)
T = np.full(N, T_skrin)
T_new = np.copy(T)

# Pole pro ukládání výsledků (Logy)
cas_plot = []
T_surf_plot = []    # Teplota na povrchu
T_core_plot = []    # Teplota ve středu
h_log = []          # Hodnota h (NOVÉ - ukládáme pro graf)
torque_log = []     # Moment (celkový přenášený)
drag_torque_log = [] # NOVÉ: Log pro vlečný moment (jen pro info)
power_log = []      # Výkon (čistý do spojky)
rpm_abs_log = []    # Log otáček (motoru)
rpm_slip_log = []   # Log otáček (prokluz)

# Proměnné pro hledání maxim
max_torque_rec = 0.0
max_power_net_rec = 0.0   # Čistý výkon do spojky
max_power_gross_rec = 0.0 # Hrubý výkon před odečtem
max_q_net_rec = 0.0
max_drag_heat_rec = 0.0   # NOVÉ: Maximální ztrátový výkon v pauze (pro výpis)

t = 0.0
step = 0

print("... Simulace běží ...")

while t < t_total:
    t_local = t % t_cyklus # Lokální čas v rámci jednoho cyklu (0 až t_cyklus)

    # Flagy pro logiku v tomto kroku
    use_pause_cooling_override = False # Zda použít Flow Limit na celou plochu
    is_drag_active = False             # Zda je aktivní vlečný moment (pro logování)
    current_drag_torque = 0.0          # Pomocná proměnná

    # ==========================================================
    # LOGIKA ČASOVÁNÍ (HILL START -> LAUNCH -> PAUSE)
    # ==========================================================
    
    # --- FÁZE 1: PŘEDZÁBĚR / ROZJEZD DO KOPCE (HOLD) ---
    if t_local < t_hold:
        # Auto stojí, motor běží, spojka prokluzuje konstantním momentem
        
        # Otáčky
        rpm_engine_abs = n_motor_hold
        rpm_slip = n_motor_hold  # Slip = Motor, protože kola stojí (0 rpm)
        
        # Moment a Výkon
        # Moment se bere z mapy motoru podle aktuálních otáček držení
        real_torque = get_torque_from_rpm(rpm_engine_abs)
        if real_torque < 0: real_torque = 0
        
        omega_slip = rpm_slip * 2 * np.pi / 60
        power_gross = real_torque * omega_slip
        
        # Chlazení: Olej teče jen drážkami
        cooling_ratio = ratio_groove 

    # --- FÁZE 2: DYNAMICKÝ ROZJEZD (LAUNCH) ---
    elif t_local < (t_hold + t_zab):
        # Auto se rozjíždí, prokluz klesá
        
        # Musíme posunout čas, aby rampa začala od 0 (relativně k začátku rozjezdu)
        t_in_launch = t_local - t_hold 
        
        # Nelineární průběh otáček (Shape Factor)
        # Pokud je exponent 1.0, je to přímka. Jinak křivka.
        ratio_linear = t_in_launch / t_zab
        ratio = ratio_linear ** RPM_SHAPE_FACTOR
        
        # Otáčky
        rpm_engine_abs = n_motor_start + (n_motor_end - n_motor_start) * ratio
        rpm_slip = n_slip_start * (1 - ratio)
        
        # Moment (z mapy motoru) a Výkon
        real_torque = get_torque_from_rpm(rpm_engine_abs)
        if real_torque < 0: real_torque = 0
        
        omega_slip = rpm_slip * 2 * np.pi / 60
        power_gross = real_torque * omega_slip
        
        # Chlazení: Olej teče jen drážkami
        cooling_ratio = ratio_groove

# --- FÁZE 3: PAUZA (COOLING & DRAG) ---
    else:
        # Spojka rozepnuta. Motor běží na volnoběh.
        rpm_engine_abs = n_motor_idle 
        
        # Prokluz v pauze:
        # Pokud auto stojí (vstup se točí, výstup stojí), je prokluz roven otáčkám motoru.
        rpm_slip = n_motor_idle     
        
        # A. VÝPOČET VLEČNÉHO MOMENTU (DRAG TORQUE)
        if ENABLE_DRAG_TORQUE:
            # Použijeme analytický model pro viskózní tření
            # Viskozitu bereme konzervativně dle teploty přívodu (T_olej_inlet)
            # (Reálně by se olej v mezeře ohřál na teplotu lamel a odpor by klesl, 
            #  použití inlet teploty dává "worst-case" vysoký odpor).
            
            M_drag = get_drag_torque_analytical(rpm_slip, T_olej_inlet, geometry_dict, h_gap_mm)
            
            real_torque = M_drag
            current_drag_torque = M_drag
            is_drag_active = True
            
            # Výkon ztracený v oleji (topí do lamel)
            omega_slip = rpm_slip * 2 * np.pi / 60
            power_gross = real_torque * omega_slip
            
            # Uložení maxima pro statistiku
            if power_gross > max_drag_heat_rec: max_drag_heat_rec = power_gross
            
        else:
            # Původní chování (vypnuto)
            real_torque = 0.0
            omega_slip = 0.0
            power_gross = 0.0

        # B. LOGIKA PRO PAUZU (OPEN CLUTCH COOLING)
        # Aktivuje se pouze pokud máme ANALYTIC režim a uživatel to povolil
        if ENABLE_OPEN_CLUTCH_COOLING and CHLAZENI_TYP == "ANALYTIC":
            cooling_ratio = ratio_pause        # Použijeme plochu pro pauzu (např. 100%)
            use_pause_cooling_override = True  # Přepneme h na Flow Limit v další sekci
        else:
            # Staré chování (stále jen drážky)
            cooling_ratio = ratio_groove

    # === APLIKACE ODBĚRU VÝKONU NÁSTAVBOU ===
    # Převedeme kW na Watty
    P_aux_W = P_auxiliary_load_kW * 1000.0
    
    # Čistý výkon pro spojku = (Moment * Skluz) - Výkon nástavby
    # POZOR: V režimu Drag Torque (Pauza) obvykle nástavba neodebírá moment z prokluzu spojky,
    # ale přímo z motoru. Zde předpokládáme, že Drag Torque jde PŘÍMO do tepla lamel.
    # Proto v pauze výkon nástavby neodečítáme od Drag Torque (ten vzniká až za motorem).
    
    if is_drag_active:
        power_net = power_gross # Vlečný moment jde celý do tepla
    else:
        power_net = power_gross - P_aux_W
        if power_net < 0: power_net = 0.0

# ==========================================================
    # VÝPOČET CHLAZENÍ (h)
    # ==========================================================
    
    # Prioritní kontrola: Jsme v pauze v režimu Open Clutch?
    if use_pause_cooling_override:
        # Použijeme speciální h pro pauzu (přepočítané na velkou plochu)
        # Toto simuluje, že lamely odskočí a olej oplachuje celou plochu.
        # Hodnota je omezena tepelnou kapacitou průtoku (aby to nebylo nekonečné).
        h_current = h_flow_limit_pause
        
    else:
        # Standardní logika pro záběr (nebo pauzu bez rozepnutí)
        if CHLAZENI_TYP == "STATIC":
            h_current = h_static_val
            
        elif CHLAZENI_TYP == "FLOW_LIMIT":
            h_current = h_flow_limit_grooves
            
        elif CHLAZENI_TYP == "NO_COOLING":
            h_current = 0.0
            
        else: # REŽIM "ANALYTIC"
            # Používáme T_olej_inlet pro viskozitu (konzervativní přístup)
            T_film_used = T_olej_inlet
                
            # Zavoláme funkci pro výpočet h 
            # (Viskozita je nyní konstantní podle T_olej_inlet, h se mění jen s RPM)
            h_current = get_cooling_analytical(rpm_engine_abs, T_film_used, geometry_dict)


    # ==========================================================
    # TEPELNÁ BILANCE (q_net)
    # ==========================================================
    
    # 1. Vstup tepla (Generation)
    # power_net obsahuje buď třecí výkon při rozjezdu, NEBO ztrátový výkon (Drag) v pauze.
    q_gen = (power_net / n_pairs / S_calc_power) * beta
    
    # 2. Odvod tepla (Cooling)
    # Rozdíl teplot je stále vůči přívodu (T_olej_inlet).
    # cooling_ratio je buď 0.06 (drážky) nebo 1.0 (celá plocha v pauze).
    q_cool = h_current * (T[0] - T_olej_inlet) * cooling_ratio
    
    # 3. Výsledek (Net Heat Flux)
    # Pokud q_gen > q_cool, teplota roste.
    # Pokud máme Drag Torque, q_gen není nula ani v pauze!
    q_net = q_gen - q_cool
    
    # Uložení maxim pro statistiku
    if real_torque > max_torque_rec: max_torque_rec = real_torque
    if power_gross > max_power_gross_rec: max_power_gross_rec = power_gross
    if power_net > max_power_net_rec: max_power_net_rec = power_net
    if q_net > max_q_net_rec: max_q_net_rec = q_net

# ==========================================================
    # SOLVER (FDM) - VÝPOČET TEPLOTY V OCELI
    # ==========================================================
    # Aktualizace materiálových vlastností pro aktuální teplotu (nelinearita)
    k_vec, cp_vec, rho_val = get_steel_props(T)
    alpha_vec = k_vec / (rho_val * cp_vec) # Teplotní difuzivita

    # A. Vnitřní uzly (vedení tepla uvnitř materiálu)
    # T_new[i] = T[i] + alpha * dt/dx^2 * (T[i+1] - 2T[i] + T[i-1])
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
 
    # B. Povrchový uzel (zde vstupuje q_net)
    # Energetická bilance na povrchu: Vstup q_net vs Vedení dovnitř
    T_new[0] = T[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
   
    # C. Středový uzel (symetrie, adiabatická stěna)
    # Gradient je nula, teplo nikam neodtéká
    T_new[-1] = T[-1] + alpha_vec[-1] * dt / dx**2 * (T[-2] - T[-1])

    # Přepis teplot pro další krok
    T[:] = T_new[:]
    t += dt
    step += 1

    # Ukládání dat pro grafy (jen každých 200 kroků, aby grafy nebyly obří)
    if step % 200 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])
        h_log.append(h_current) # Ukládáme aktuální h
        
        torque_log.append(real_torque)
        # NOVÉ: Ukládáme vlečný moment zvlášť pro kontrolu
        # Pokud není aktivní, ukládáme 0
        if is_drag_active:
            drag_torque_log.append(current_drag_torque)
        else:
            drag_torque_log.append(0.0)
            
        power_log.append(power_net) # Ukládáme čistý výkon (zdroj tepla)
        rpm_abs_log.append(rpm_engine_abs)
        rpm_slip_log.append(rpm_slip)

# ==============================================================================
# ==============================================================================
#                       5. VÝPIS A GRAFY VÝSLEDKŮ
# ==============================================================================
# ==============================================================================

print("\n" + "="*60)
print(f" VÝSLEDKY SIMULACE")
print("="*60)
print(f"Koeficient Beta:             {beta:.3f} (-)")
print(f"Plocha lamely (S_calc):      {S_calc_power * 10000:.2f} cm² ({area_note})")
print(f"Hydraulický průměr drážky:   {Dh*1000:.2f} mm")
print("-" * 60)
print(f"Špičkový točivý moment:      {max_torque_rec:.1f} Nm")
print(f"1. CELKOVÝ TŘECÍ VÝKON (GROSS):  {max_power_gross_rec / 1000:.1f} kW")
print(f"   (Teoretické maximum bez odběru)")

# Výpis pro Drag Torque (pokud byl aktivní)
if ENABLE_DRAG_TORQUE and max_drag_heat_rec > 0:
    print(f"   -> Z toho VLEČNÝ MOMENT (Drag): {max_drag_heat_rec:.1f} W ({max_drag_heat_rec/1000:.2f} kW)")
    print(f"      (Toto teplo se tvoří v pauze třením oleje!)")
else:
    print(f"   -> Vlečný moment:             0.0 W (Deaktivováno nebo zanedbatelné)")

print(f"2. ODBĚR NÁSTAVBOU:              {P_auxiliary_load_kW:.1f} kW")
print(f"3. VÝSLEDNÝ VÝKON DO SPOJKY:     {max_power_net_rec / 1000:.1f} kW")
print(f"   (Poníženo o nástavbu)")
print("-" * 60)
print(f"Špičkový tepelný tok (q):    {max_q_net_rec / 1e6:.2f} MW/m²")
print(f"MAXIMÁLNÍ TEPLOTA POVRCHU:   {max(T_surf_plot):.1f} °C")
print("="*60)

# Vytvoření okna se 4 GRAFY pod sebou
# figsize zvětšena na (12, 16), aby se 4 grafy vešly a byly čitelné
fig, (ax1, ax3, ax5, ax7) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# --- GRAF 1: TEPLOTA MATERIÁLU ---
# Zde uvidíte vliv Drag Torque: Teplota v pauze nebude klesat tak rychle,
# nebo se může dokonce ustálit nad teplotou oleje (rovnovážný stav).
ax1.plot(cas_plot, T_surf_plot, 'r-', label='Povrch Oceli (Styk s obložením)', linewidth=1.5)
ax1.plot(cas_plot, T_core_plot, 'b--', label='Střed Oceli (Symetrie)', linewidth=1.5)
ax1.set_ylabel('Teplota [°C]', fontsize=12, fontweight='bold')
ax1.set_title(f'1. Průběh teploty ocelové lamely\n(Režim: {CHLAZENI_TYP}, Drag Torque: {"ZAP" if ENABLE_DRAG_TORQUE else "VYP"})', fontsize=14)
ax1.grid(True, alpha=0.5)
ax1.legend(loc='upper right', fontsize=10)

# --- GRAF 2: ZÁTĚŽ (MOMENT A VÝKON) ---
ax3.set_ylabel('Moment [Nm]', color='green', fontsize=12, fontweight='bold')

# Vykreslení celkového momentu
line1 = ax3.plot(cas_plot, torque_log, 'g-', label='Moment (Motor / Drag)', alpha=0.8)

# Pokud je aktivní Drag Torque, můžeme ho zvýraznit (volitelné, zde je zahrnut v torque_log)
# Ale pro přehlednost necháme hlavní křivku.

ax3.tick_params(axis='y', labelcolor='green')
ax3.grid(True, alpha=0.5)

# Druhá osa Y pro výkon
ax4 = ax3.twinx()
ax4.set_ylabel('Čistý Výkon [kW]', color='orange', fontsize=12, fontweight='bold')
power_kw = [p / 1000 for p in power_log]

# Zde uvidíte, že v pauze výkon není nula (pokud je Drag Torque aktivní)
line2 = ax4.plot(cas_plot, power_kw, color='orange', linestyle='--', label=f'Tepelný výkon (kW)')
ax4.tick_params(axis='y', labelcolor='orange')

# Společná legenda
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc='center right', fontsize=10)
ax3.set_title('2. Zatížení spojky (Vlečný moment je vidět v pauze)', fontsize=12)

# --- GRAF 3: PRŮBĚH OTÁČEK ---
# Zde kontrolujeme, zda sedí fáze Hold, Launch a Pauza
ax5.set_ylabel('Otáčky [rpm]', color='black', fontsize=12, fontweight='bold')

# Vykreslení otáček motoru (Absolutní)
ax5.plot(cas_plot, rpm_abs_log, 'k-', label='Otáčky Motoru (Koš spojky)', linewidth=2)

# Vykreslení otáček prokluzu (Rozdíl rychlostí)
# Všimněte si: V pauze je prokluz roven otáčkám motoru (protože výstup stojí),
# což je důvod, proč vzniká Drag Torque.
ax5.plot(cas_plot, rpm_slip_log, 'm--', label='Otáčky Prokluzu (Delta n)', linewidth=1.5)

ax5.set_title('3. Průběh otáček (Zdrojem vlečného momentu je prokluz v pauze)', fontsize=12)
ax5.grid(True, alpha=0.5)
ax5.legend(loc='upper right', fontsize=10)

# --- NOVÉ: GRAF 4 - SOUČINITEL PŘESTUPU TEPLA (h) + OTÁČKY NA POZADÍ ---
ax7.set_xlabel('Čas simulace [s]', fontsize=12, fontweight='bold')
ax7.set_ylabel('h [W/m2K]', color='blue', fontsize=12, fontweight='bold')

# Hlavní křivka (h)
# Pokud je aktivní Open Clutch Cooling, uvidíte zde skokovou změnu v pauze
l_h = ax7.plot(cas_plot, h_log, 'c-', label='Součinitel přestupu tepla h', linewidth=2)
ax7.fill_between(cas_plot, 0, h_log, color='cyan', alpha=0.1) 
ax7.tick_params(axis='y', labelcolor='blue')

# Sekundární osa pro otáčky (aby byla vidět závislost)
ax8 = ax7.twinx()
ax8.set_ylabel('Otáčky Motoru [rpm]', color='grey', fontsize=10)
l_rpm = ax8.plot(cas_plot, rpm_abs_log, color='grey', linestyle=':', label='Otáčky motoru', alpha=0.5)
ax8.tick_params(axis='y', labelcolor='grey')

# Legenda
lns2 = l_h + l_rpm
labs2 = [l.get_label() for l in lns2]
ax7.legend(lns2, labs2, loc='upper right', fontsize=10)

title_text = '4. Intenzita chlazení (Závislost h na otáčkách)'
if ENABLE_OPEN_CLUTCH_COOLING:
    title_text += '\n(V pauze aktivní zvýšené chlazení rozlepených lamel)'
ax7.set_title(title_text, fontsize=12)
ax7.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# ==============================================================================
# KONEC SIMULACE
# ==============================================================================
