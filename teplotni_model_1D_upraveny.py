import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


"""
================================================================================
NÁZEV: 1D TERMOHYDRAULICKÝ MODEL MOKRÉ SPOJKY (FDM/MKP)
VERZE: 2.1 (S Drag Torque & Laminárním/Turbulentním přepínáním)
================================================================================

POPIS MODELU:
Tento skript slouží k detailní simulaci teplotního chování ocelové (separační) 
lamely mokré spojky v čase. Na rozdíl od jednoduchých 0D modelů (které uvažují 
lamelu jako jeden hmotný bod s rovnoměrnou teplotou) využívá tento model 
numerickou metodu konečných diferencí (FDM) k výpočtu teplotního gradientu 
napříč tloušťkou materiálu.

HLAVNÍ ROZDÍLY OPROTI 0D MODELU:
--------------------------------
1. Rozložení teploty (Gradient):
   - Model rozlišuje teplotu na POVRCHU (styková plocha s třecím obložením) 
     a v JÁDŘE (střed materiálu).
   - Dokáže odhalit krátkodobé teplotní špičky na povrchu ("Skin effect"), 
     které 0D model zprůměruje a skryje.

2. Pokročilá Hydrodynamika (Chlazení):
   - Součinitel přestupu tepla 'h' není konstanta.
   - Počítá se dynamicky v každém kroku na základě otáček, geometrie drážek 
     a viskozity oleje.
   - Režim 'ANALYTIC_REAL' automaticky přepíná mezi Laminárním a Turbulentním 
     prouděním podle Reynoldsova čísla (Re).

3. Vlečný moment (Drag Torque):
   - Zohledňuje parazitní zahřívání v rozpojeném stavu (Pauza).
   - Počítá viskózní tření oleje mezi lamelami (kritické pro DCT převodovky 
     a opakované cykly, kde teplota neklesá k nule).

FYZIKÁLNÍ PRINCIP VÝPOČTU:
--------------------------
1. ZDROJ TEPLA: 
   Třecí výkon (Moment x Skluz) se poníží o koeficient Beta (dělení tepla 
   mezi ocel a obložení). Do oceli vstupuje cca 90-95 % tepla, zbytek jde 
   do izolantu (papíru).

2. VEDENÍ TEPLA (Conduction): 
   Řeší se 1D rovnice vedení tepla napříč tloušťkou oceli.

3. ODVOD TEPLA (Convection): 
   Povrch je ochlazován olejem. Intenzita chlazení je omezena buď fyzikou 
   mezní vrstvy (Nusseltovo číslo), nebo celkovým průtokem oleje (Flow Limit).

VSTUPY A PŘEDPOKLADY:
---------------------
- Materiál: Nelineární vlastnosti oceli (vodivost a kapacita se mění s teplotou).
- Geometrie: Uvažuje se efektivní plocha po odečtení drážek (volitelné).
- Okrajové podmínky: Adiabatická stěna ve středu lamely (symetrie), 
  teplo uniká pouze do oleje (zanedbáno vedení do hřídele -> konzervativní).

VÝSTUPY:
--------
- Časový průběh teploty povrchu a jádra.
- Celková tepelná bilance a potřebný chladicí výkon.
- Identifikace rizika spálení oleje (povrchová teplota) vs. přehřátí materiálu (objemová teplota).

================================================================================
"""

# NASTAVENÍ MODELU
# --------------------------------------------------------------------------------
# řádek 312 - Parametry spojky

# podávat na  250?
# TYP OCHLAZOVÁNÍ SPOJKY
# MOŽNOSTI:
#   "ANALYTIC_REAL" = (NOVÉ) Dynamicky přepíná Laminární/Turbulentní tok dle otáček (Re).
#   "ANALYTIC"      = Původní dynamický výpočet (používá pouze turbulentní korelaci).
#   "FLOW_LIMIT"    = Počítá maximální možné chlazení dle kapacity průtoku.
#   "NO_COOLING"    = Vypne chlazení (h = 0).

CHLAZENI_TYP = "ANALYTIC"   # <--- ZMĚNA MOŽNOSTI
# řádek 209 - korekce teplotního součinitele h
# řádek 248 - změna Nu pro laminární proudění

# Započítání zmenšené plochy obložení, kterou odebírájí drážky (Waffle profil)
# True = Větší tepelný tok do separační lamely (menší plocha obložení)
# False = Ignoruje se (používá se celá plocha obložení)
# řádek 292 - modifikace plochy drážek 
INCLUDE_AREA_REDUCTION = False  

# Výkon, který odebírá PTO
P_auxiliary_load_kW = 100.0  # [kW] 

# Teplota oleje na vstupu (určuje chlazení i viskozitu).
T_olej_inlet = 70.0   # [°C]


# Pro předzáběr spojky v kopci
# True = Povolit předzáběr
# False = Vypnout předzáběr
ENABLE_HILL_START = False   

# Parametry pro pro předzáběr
t_hold       = 1.0    # [s]   Doba trvání předzáběru ()
n_motor_hold = 800.0  # [rpm] Otáčky motoru při předzáběru

# Tvar otáček motoru během záběru (průběh spínání spojky)
# Exponent = 1.0 (Lineární), Exponent > 1.0 (Konkávní - Jízda na spojce), Exponent < 1.0 (Konvexní - Rychlé sepnutí)
RPM_SHAPE_FACTOR = 1.0 

# OTÁČKY MOTORU (Absolutní)
n_motor_start = 1200.0  # [rpm] Start rozjezdu (ovlivňuje chlazení)
n_motor_end   = 1200.0  # [rpm] Konec prokluzu (ovlivňuje chlazení)
n_motor_idle  = 1200.0   # [rpm] Otáčky motoru v rozepnutém stavu (pro výpočet unášivého momentu a chlazení)

# Rozdíl otáček "motor <-> převodovka" (Relativní rozdíl)
n_slip_start  = 1200.0  # [rpm] Počáteční prokluz (počátek záběru)
n_slip_end    = 0.0     # [rpm] Konec prokluzu (konec záběru)

# NASTAVENÍ CYKLU
n_cyklu = 8           # Počet opakování (POČET ROZBĚHŮ)
t_zab   = 1.74        # [s] Doba prokluzu 
t_pauza = 30.0         # [s] Doba rozpojení spojky

# CHOVÁNÍ V DOBĚ ROZPOJENÍ (OPEN CLUTCH & DRAG TORQUE) 

# CHLAZENÍ SPOJKY V DOBĚ ROZPOJENÍ
# True = Povolit chlazení celé plochy v rozpojeném stavu
# False = Vypnout chlazení celé plochy v rozpojeném stavu
ENABLE_OPEN_CLUTCH_COOLING = False  # Povolit "Flow Limit" na celou plochu v pauze
ratio_pause = 1.0                  # 100% plochy se chladí

# Unašivý moment
# True = Počítat odporový moment během rozpojení
# False = Ignorovat odporový moment během rozpojení
ENABLE_DRAG_TORQUE = True    
h_gap_mm = 0.2               # [mm] Vůle mezi lamelou a diskem

# Ošetření logiky pro začatek v nule při předzáběru
if not ENABLE_HILL_START:
    t_hold = 0.0

#DEFINICE FYZIKÁLNÍCH FUNKCÍ
# -----------------------------------------------------------------------------
def get_steel_props(T_celsius):
    """
    Vypočítá materiálové vlastnosti separační lamely v závislosti na teplotě
    """
    # Oříznutí teploty
    T = np.clip(T_celsius, 20.0, 1000.0)
    
    # Tepelná vodivost k [W/m.K] - Schopnost vést teplo.
    k = 54.0 - 0.028 * T
    
    # 2. Měrná tepelná kapacita cp [J/kg.K]
    c_p = 450.0 + 0.28 * T
    
    # 3. Hustota rho [kg/m3] - Hmotnost objemu. Považujeme za konstantu.
    rho = 7850.0
    
    return k, c_p, rho

def get_cooling_analytical(rpm, T_viscosity_input, geometry):
    """
    Vstupy:
      rpm               : Aktuální otáčky koše spojky (motoru)
      T_viscosity_input : Teplota použitá PRO URČENÍ VISKOZITY.
      geometry          : Slovník s rozměry (poloměry, hydraulický průměr)
    """
    # Ošetření nulových otáček
    if rpm < 10: return 50.0 
    
    # FYZIKÁLNÍ VLASTNOSTI OLEJE
    rho = 850.0       # Hustota [kg/m3]
    lam_oil = 0.14    # Tepelná vodivost oleje (lambda) [W/m.K]
    c_oil = 2000.0    # Tepelná kapacita oleje (cp) [J/kg.K]
    
    # Kinematická viskozita (nu) [m2/s] (dle zadané teploty)
    nu = np.interp(T_viscosity_input, [40, 100], [30e-6, 7e-6]) 
    
    # GEOMETRIE
    r_in = geometry['r_in']
    r_out = geometry['r_out']
    Dh = geometry['Dh'] # Hydraulický poloměr (d_h)
    
    # VÝPOČET DLE HYDRODYNAMIK
    
    # Úhlová rychlost [rad/s]
    omega = rpm * (2 * np.pi / 60)
    
    # Rychlost toku v drážce (v_r)
    v_oil = omega * np.sqrt(r_out**2 - r_in**2)
    
    # Reynoldsovo číslo (Re)
    Re = (v_oil * Dh) / nu
    
    # Prandtlovo číslo (Pr)
    Pr = (rho * c_oil * nu) / lam_oil
    
    # Nusseltovo číslo (Nu)
    # Dittus-Boelterova korelace pro turbulentní tok
    Nu = 0.023 * (Re**0.8) * (Pr**0.3)
    
    # Výpočet součinitele přestupu tepla h [W/m2.K]
    h_pipe = (Nu / Dh) * lam_oil
    
    # Korekce pro zvýšení přestupu tepla (drsnost apod.)
    enhancement_factor = 1.0 
    
    h = h_pipe * enhancement_factor
    
    return h

def get_cooling_analytic_real(rpm, T_viscosity_input, geometry):
    """
    Režim proudění (Laminární vs Turbulentní) podle Reynoldsova čísla.
    """
    # Ošetření nulových otáček
    if rpm < 10: return 50.0 
    
    # FYZIKÁLNÍ VLASTNOSTI OLEJE
    rho = 850.0       
    lam_oil = 0.14    
    c_oil = 2000.0    
    
    # Kinematická viskozita
    nu = np.interp(T_viscosity_input, [40, 100], [30e-6, 7e-6]) 
    
    # GEOMETRIE
    r_in = geometry['r_in']
    r_out = geometry['r_out']
    Dh = geometry['Dh'] 
    
    # VÝPOČET REYNOLDSOVA ČÍSLA
    omega = rpm * (2 * np.pi / 60)
    v_oil = omega * np.sqrt(r_out**2 - r_in**2) # Rychlost v drážce
    Re = (v_oil * Dh) / nu
    
    # Prandtlovo číslo
    Pr = (rho * c_oil * nu) / lam_oil
    

    # Určení proudění a výpočet Nusseltova čísla
    if Re < 2300:
        # LAMINÁRNÍ TOK
        # Nusseltovo číslo pro laminární proudění v obdélníkovém průřezu (lze změnit)
        Nu = 6.08 
    else:
        # TURBULENTNÍ TOK (Dittus-Boelter)
        Nu = 0.023 * (Re**0.8) * (Pr**0.3)
    
    # Výpočet h
    h = (Nu / Dh) * lam_oil

    return h

def get_drag_torque_analytical(rpm_slip, T_viscosity_input, geometry, h_gap_mm):
    """
    Výpočet unašivého momentu způsobeného viskozitou oleje
    Vzorec: M = (pi * mu * omega * (R_out^4 - R_in^4)) / (2 * h_gap)
    
    Vstupy:
      rpm_slip          : Rozdíl otáček mezi motorem a převodovkou (prokluz)
      T_viscosity_input : Teplota pro určení viskozity (zde T_olej_inlet)
      geometry          : Slovník s poloměry
      h_gap_mm          : Vůle mezi lamelami v mm
    """
    # Viskozita oleje (stejná interpolace jako u chlazení)
    rho = 850.0
    # Interpolace kinematické viskozity (nu) [m2/s]
    nu = np.interp(T_viscosity_input, [40, 100], [30e-6, 7e-6]) 
    # Výpočet dynamické viskozity (mu) [Pa.s]
    mu = nu * rho 
    
    # Geometrie a převody jednotek
    r_out = geometry['r_out']
    r_in = geometry['r_in']
    h_gap = h_gap_mm / 1000.0 # Převod z mm na metry [m]
    
    # Úhlová rychlost prokluzu [rad/s]
    omega_slip = rpm_slip * (2 * np.pi / 60)
    
    # Výpočet momentu
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
        print(f"CHYBA: Soubor '{filename}' nenalezen! (použity hodnoty níže)")
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
    except KeyError:
        print("CHYBA: Excel musí mít sloupce 'RPM' a 'Torque'.")
        raise
    # interpolace
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data

# PARAMETRY A GEOMETRIE
# -----------------------------------------------------------------------------

# Načtení mapy motoru
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()

# MATERIÁLOVÉ KONSTANTY
# Referenční hodnoty pro výpočet koeficientu Beta (rozdělení tepla mezi ocel(sep. lamela) a obložením (obložena lamela)
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2 # Třecí obložení (papír)

# Koeficient Beta - Kolik % tepla jde do oceli? 
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

# Vlastnosti oleje pro chlazení
c_oil = 2000.0       # Kapacita oleje
lambda_oil = 0.14    # Vodivost
q_total_lmin = 6.0 # L/min (pro režim chlazení FLOW LIMIT)

# GEOMETRIE SPOJKY
n_pairs = 14         # Počet třecích ploch
r_out = 0.124        # [m] Vnější poloměr
r_in = 0.0875        # [m] Vnitřní poloměr
tloustka_oceli = 0.004 # [m] Tloušťka ocelové separační lamely

T_skrin = T_olej_inlet # Počáteční teplota

# GEOMETRIE DRÁŽEK (WAFFLE)
sirka_drazky = 0.0015  # [m]
hloubka_drazky = 0.0002 # [m]
roztec_drazek = 0.009  # [m]

# Výpočet hydraulického průměr (Dh) - "Efektivní průměr trubky (drážky)"
S_tok = sirka_drazky * hloubka_drazky
O_tok = 2 * (sirka_drazky + hloubka_drazky)
Dh = 4 * S_tok / O_tok

geometry_dict = {'r_in': r_in, 'r_out': r_out, 'Dh': Dh}

# PLOCHY A REDUKCE
S_celkova_mezikruzi = np.pi * (r_out**2 - r_in**2)
ratio_groove = 0.06  # Kolik % plochy zabírají drážky (nejlépe z 3D dat)
S_cooling = S_celkova_mezikruzi * ratio_groove          # Plocha kudy teče olej
S_contact = S_celkova_mezikruzi * (1 - ratio_groove)    # Skutečná plocha kam jde teplo (obložení-ocel)

# Aplikace volby (INCLUDE_AREA_REDUCTION)
if INCLUDE_AREA_REDUCTION:
    S_calc_power = S_contact 
    area_note = "Redukovaná (Odečteny drážky)"
else:
    S_calc_power = S_celkova_mezikruzi 
    area_note = "Celková (Ignorovány drážky)"



# Výpočet h pro různé režimy chlazení
mdot_per_surface = (q_total_lmin / 60 / 1000 * 850) / n_pairs 

# Flow Limit pro Ddrážky
h_flow_limit_grooves = (mdot_per_surface * c_oil) / S_cooling

# Flow Limit pro rozepnuto

S_cooling_pause = S_celkova_mezikruzi * ratio_pause
h_flow_limit_pause = (mdot_per_surface * c_oil) / S_cooling_pause

# Analytické h bez modifikace
h_real_start_demo = get_cooling_analytic_real(n_motor_start, T_olej_inlet, geometry_dict)

# Analytické h s modifikací
h_analytic_demo_val = get_cooling_analytical(n_motor_start, T_olej_inlet, geometry_dict)

# Výpis info do konzole
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
print(f"  1. ANALYTIC_REAL (Start):  {h_real_start_demo:.1f} W/m2K (Laminární/Turbulentní dle otáček)")
print(f"  2. FLOW LIMIT (Drážky):    {h_flow_limit_grooves:.1f} W/m2K (při {q_total_lmin} L/min)")
print(f"  3. ANALYTIC (při {n_motor_start:.0f} rpm): {h_analytic_demo_val:.1f} W/m2K (Vždy turbulentní)")
print(f"  4. NO COOLING:             0.0 W/m2K")
print("-" * 60)


#  HLAVNÍ SIMULAČNÍ SMYČKA (SOLVER)


# Celkový čas cyklu se prodlouží o dobu držení v kopci (t_hold)
t_cyklus = t_hold + t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Nastavení sítě
# Tloušťka lamely podělena N, dále využití symetrii
L = tloustka_oceli / 2 # symetrie
N = 50                 # Počet uzlů
dx = L / (N - 1)       # Vzdálenost mezi uzly [m]

# Časový krok (dt)
k_c, c_c, rho_c = get_steel_props(20.0)
dt = 0.9 * (0.5 * dx**2 / (k_c / (rho_c * c_c))) # Courantovo kritérium

# Inicializace teplotního pole 
T = np.full(N, T_skrin)
T_new = np.copy(T)

# Pole pro ukládání výsledků 
cas_plot = []
T_surf_plot = []    # Teplota na povrchu
T_core_plot = []    # Teplota ve středu
h_log = []          # Hodnota h
torque_log = []     # Moment (celkový přenášený)
drag_torque_log = [] # Log pro unašivý moment
power_log = []      # Výkon (čistý do spojky)
rpm_abs_log = []    # Log otáček (motoru)
rpm_slip_log = []   # Log otáček (prokluz)

# Proměnné pro hledání maxim
max_torque_rec = 0.0
max_power_net_rec = 0.0   # Čistý výkon do spojky
max_power_gross_rec = 0.0 # Hrubý výkon před odečtem
max_q_net_rec = 0.0
max_drag_heat_rec = 0.0   #  Maximální ztrátový výkon v pauze (pro výpis)

t = 0.0
step = 0

print("... Simulace běží ...")

while t < t_total:
    t_local = t % t_cyklus # Lokální čas v rámci jednoho cyklu (0 až t_cyklus)

    use_pause_cooling_override = False
    is_drag_active = False             
    current_drag_torque = 0.0          

    
    # LOGIKA  (PŘEDZÁBĚR -> ROZJEZD -> ROZPOJENÁ SPOJKA)

    
    # PŘEDZÁBĚR 
    if t_local < t_hold:
             
        # Otáčky
        rpm_engine_abs = n_motor_hold
        rpm_slip = n_motor_hold  
        
        # Moment a Výkon
        # Moment se bere z mapy motoru podle aktuálních otáček držení
        real_torque = get_torque_from_rpm(rpm_engine_abs)
        if real_torque < 0: real_torque = 0
        
        omega_slip = rpm_slip * 2 * np.pi / 60
        power_gross = real_torque * omega_slip
        
        # Chlazení: Olej teče jen drážkami
        cooling_ratio = ratio_groove 

    # ROZJEZD SPOJKY
    elif t_local < (t_hold + t_zab):
                
       
        t_in_launch = t_local - t_hold 
        
        # Nelineární průběh otáček
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

# ROZPOJENÁ SPOJK
    else:
        # Spojka rozepnuta
        rpm_engine_abs = n_motor_idle 
        
        # Prokluz v pauze:
        rpm_slip = n_motor_idle     
        
        # VÝPOČET UNAŠIVÉHO MOMENTU
        if ENABLE_DRAG_TORQUE:
          
            
            M_drag = get_drag_torque_analytical(rpm_slip, T_olej_inlet, geometry_dict, h_gap_mm)
            
            real_torque = M_drag
            current_drag_torque = M_drag
            is_drag_active = True
            
            # Výkon ztracený v oleji
            omega_slip = rpm_slip * 2 * np.pi / 60
            power_gross = real_torque * omega_slip
            
            if power_gross > max_drag_heat_rec: max_drag_heat_rec = power_gross
            
        else:
            real_torque = 0.0
            omega_slip = 0.0
            power_gross = 0.0

        # ROZPOJENÁ SPOJKA - CHLAZENÍ
        if ENABLE_OPEN_CLUTCH_COOLING and CHLAZENI_TYP == "ANALYTIC":
            cooling_ratio = ratio_pause        
            use_pause_cooling_override = True  
        else:
            cooling_ratio = ratio_groove


    P_aux_W = P_auxiliary_load_kW * 1000.0
    
    # Čistý výkon pro spojku = (Moment * Skluz) - Výkon nástavby
    
    if is_drag_active:
        power_net = power_gross # Vlečný moment jde celý do tepla
    else:
        power_net = power_gross - P_aux_W
        if power_net < 0: power_net = 0.0


    # VÝPOČET CHLAZENÍ (h)
    
    # ==========================================================
    # VÝPOČET CHLAZENÍ (h)
    # ==========================================================
    
    # Prioritní přepínač pro režim chlazení v pauze
    if use_pause_cooling_override:
        # Chlazení v rozpojeno
        h_current = h_flow_limit_pause
        
    else:
        # Standardní režimy chlazení během aktivního záběru
        if CHLAZENI_TYP == "ANALYTIC_REAL":
            T_film_used = T_olej_inlet
            h_current = get_cooling_analytic_real(rpm_engine_abs, T_film_used, geometry_dict)
            
        elif CHLAZENI_TYP == "FLOW_LIMIT":
            h_current = h_flow_limit_grooves
            
        elif CHLAZENI_TYP == "NO_COOLING":
            h_current = 0.0
            
        else: # REŽIM "ANALYTIC" (Default)
            T_film_used = T_olej_inlet
            # Viskozita je nyní konstantní podle T_olej_inlet, h se mění jen s RPM
            h_current = get_cooling_analytical(rpm_engine_abs, T_film_used, geometry_dict)


    
    # TEPELNÁ BILANCE (q_net)
    
    
    # Vstup tepla (Generation)
    q_gen = (power_net / n_pairs / S_calc_power) * beta
    
    # Odvod tepla (ochalzování)
    q_cool = h_current * (T[0] - T_olej_inlet) * cooling_ratio
    
    # Výsledek (Net Heat Flux)
    q_net = q_gen - q_cool
    
    # Uložení maxim pro statistiku
    if real_torque > max_torque_rec: max_torque_rec = real_torque
    if power_gross > max_power_gross_rec: max_power_gross_rec = power_gross
    if power_net > max_power_net_rec: max_power_net_rec = power_net
    if q_net > max_q_net_rec: max_q_net_rec = q_net

    # SOLVER (FDM) - VÝPOČET TEPLOTY V OCELI
    k_vec, cp_vec, rho_val = get_steel_props(T)
    alpha_vec = k_vec / (rho_val * cp_vec) # Teplotní difuzivita

    # Vnitřní uzly (vedení tepla uvnitř materiálu)
    # T_new[i] = T[i] + alpha * dt/dx^2 * (T[i+1] - 2T[i] + T[i-1])
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
 
    # Povrchový uzel (zde vstupuje q_net) 
    T_new[0] = T[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
   
    # Středový uzel (symetrie, adiabatická stěna)
    # Gradient je nula, teplo nikam neodtéká
    T_new[-1] = T[-1] + alpha_vec[-1] * dt / dx**2 * (T[-2] - T[-1])

    # Přepis teplot pro další krok
    T[:] = T_new[:]
    t += dt
    step += 1

    # Ukládání dat pro grafy 
    if step % 200 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])
        h_log.append(h_current) # Ukládáme aktuální h
        
        torque_log.append(real_torque)
        
        if is_drag_active:
            drag_torque_log.append(current_drag_torque)
        else:
            drag_torque_log.append(0.0)
            
        power_log.append(power_net) # čistý výkon (zdroj tepla)
        rpm_abs_log.append(rpm_engine_abs)
        rpm_slip_log.append(rpm_slip)


# VÝPIS A GRAFY VÝSLEDKŮ

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

# Výpis pro unašivý moment
if ENABLE_DRAG_TORQUE and max_drag_heat_rec > 0:
    print(f"   -> Z toho VLEČNÝ MOMENT (Drag): {max_drag_heat_rec:.1f} W ({max_drag_heat_rec/1000:.2f} kW)")
    print(f"      -")
else:
    print(f"   -> Vlečný moment:             0.0 W ")

print(f"2. ODBĚR NÁSTAVBOU:              {P_auxiliary_load_kW:.1f} kW")
print(f"3. VÝSLEDNÝ VÝKON DO SPOJKY (neponížen o beta):     {max_power_net_rec / 1000:.1f} kW")
print(f"   (Poníženo o nástavbu)")
print("-" * 60)
print(f"Špičkový tepelný tok (q):    {max_q_net_rec / 1e6:.2f} MW/m²")
print(f"MAXIMÁLNÍ TEPLOTA POVRCHU:   {max(T_surf_plot):.1f} °C")
print("="*60)


fig, (ax1, ax3, ax5, ax7) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# GRAF 1: TEPLOTA MATERIÁLU 

ax1.plot(cas_plot, T_surf_plot, 'r-', label='Povrch Oceli (Styk s obložením)', linewidth=1.5)
ax1.plot(cas_plot, T_core_plot, 'b--', label='Střed Oceli (Symetrie)', linewidth=1.5)
ax1.set_ylabel('Teplota [°C]', fontsize=12, fontweight='bold')
ax1.set_title(f'1. Průběh teploty ocelové lamely\n(Režim: {CHLAZENI_TYP}, Drag Torque: {"ZAP" if ENABLE_DRAG_TORQUE else "VYP"})', fontsize=14)
ax1.grid(True, alpha=0.5)
ax1.legend(loc='upper right', fontsize=10)

# GRAF 2: ZÁTĚŽ (MOMENT A VÝKON)
ax3.set_ylabel('Moment [Nm]', color='green', fontsize=12, fontweight='bold')

# Vykreslení celkového momentu
line1 = ax3.plot(cas_plot, torque_log, 'g-', label='Moment (Motor / Drag)', alpha=0.8)


ax3.tick_params(axis='y', labelcolor='green')
ax3.grid(True, alpha=0.5)

# Druhá osa Y - VÝKON
ax4 = ax3.twinx()
ax4.set_ylabel('Čistý Výkon [kW]', color='orange', fontsize=12, fontweight='bold')
power_kw = [p / 1000 for p in power_log]

line2 = ax4.plot(cas_plot, power_kw, color='orange', linestyle='--', label=f'Tepelný výkon (kW)')
ax4.tick_params(axis='y', labelcolor='orange')

# Společná legenda
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc='center right', fontsize=10)
ax3.set_title('2. Zatížení spojky (Vlečný moment je vidět v pauze)', fontsize=12)

# PRŮBĚH OTÁČEK
ax5.set_ylabel('Otáčky [rpm]', color='black', fontsize=12, fontweight='bold')

# Vykreslení otáček motoru (Absolutní)
ax5.plot(cas_plot, rpm_abs_log, 'k-', label='Otáčky Motoru (Koš spojky)', linewidth=2)

# Vykreslení otáček prokluzu (Rozdíl rychlostí)
ax5.plot(cas_plot, rpm_slip_log, 'm--', label='Otáčky Prokluzu (Delta n)', linewidth=1.5)

ax5.set_title('3. Průběh otáček (Zdrojem vlečného momentu je prokluz v pauze)', fontsize=12)
ax5.grid(True, alpha=0.5)
ax5.legend(loc='upper right', fontsize=10)

# GRAF 4 - SOUČINITEL PŘESTUPU TEPLA (h)
ax7.set_xlabel('Čas simulace [s]', fontsize=12, fontweight='bold')
ax7.set_ylabel('h [W/m2K]', color='blue', fontsize=12, fontweight='bold')

# Hlavní křivka (h)
l_h = ax7.plot(cas_plot, h_log, 'c-', label='Součinitel přestupu tepla h', linewidth=2)
ax7.fill_between(cas_plot, 0, h_log, color='cyan', alpha=0.1) 
ax7.tick_params(axis='y', labelcolor='blue')

# Sekundární osa pro otáčky 
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

# 
# KONEC SIMULACE
