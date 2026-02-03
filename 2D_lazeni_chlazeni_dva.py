# ==============================================================================
# ČÁST 1: IMPORTY A UŽIVATELSKÉ NASTAVENÍ (KONFIGURACE)
# ==============================================================================
# Zde načítáme "knihovny" - balíčky funkcí, které Python potřebuje pro matematiku.

import numpy as np                  # "NumPy" - Hlavní balík pro matice a vektory
import matplotlib.pyplot as plt     # "Matplotlib" - Balík pro kreslení grafů
import pandas as pd                 # "Pandas" - Pro práci s tabulkami (Excel)
from scipy.interpolate import interp1d # Funkce pro spojování bodů (interpolace)

# ==============================================================================
#                       1. UŽIVATELSKÉ NASTAVENÍ
# ==============================================================================
# Tady jsou "čudlíky", kterými ovládáte celou simulaci.

# --- A. NASTAVENÍ REŽIMU CHLAZENÍ ---
# Zde vybíráte, jak se má chovat olej.
#
# NOVÁ LOGIKA PRO VŠECHNY REŽIMY "ANALYTIC":
# Průtok už se nepočítá z místní rychlosti kola, ale z GLOBÁLNÍHO TLAKOVÉHO ROZDÍLU
# (odstředivý tlak mezi vstupem a výstupem). To je fyzikálně správně.
#
# 1. "ANALYTIC_FIXED" 
#    - Průtok je NATVRDO nastavený (např. externí čerpadlo).
#    - Olej se cestou OHŘÍVÁ (reálné chování).
#
# 2. "ANALYTIC_RPM" (Samonasávání - REALITA)
#    - Průtok vzniká ODSTŘEDIVOU SILOU motoru (čím vyšší otáčky, tím větší tlak a průtok).
#    - Olej se cestou OHŘÍVÁ (reálné chování).
#    - Toto simuluje skutečné auto bez čerpadla.
#
# 3. "ANALYTIC_FIX_TEMPERATURE" (Samonasávání - VALIDACE)
#    - Průtok vzniká stejně jako v bodě 2 (ODSTŘEDIVÁ SÍLA).
#    - ALE: Olej se NEOHŘÍVÁ. V každém bodě lamely má stále vstupní teplotu.
#    - Tento režim slouží pro kontrolu výpočtů oproti jednoduchým vzorcům.
#
# 4. "NO_COOLING"
#    - Žádné chlazení. Spojka se jen peče ve vlastní šťávě.

CHLAZENI_TYP = "NO_COOLING"  # <--- ZDE ZVOLTE REŽIM

# Započítat zmenšení plochy o drážky? 
# True = Plocha pro odvod tepla je menší o plochu drážek (konzervativní).
# False = Počítáme celou plochu mezikruží.
INCLUDE_AREA_REDUCTION = False

# --- B. ODBĚR VÝKONU (PTO / Hydraulika) ---
# Kolik kW sebere čerpadlo nástavby motoru ještě před spojkou.
# Pokud čerpadlo bere 10 kW a motor dává 15 kW, spojka přenáší jen 5 kW.
P_auxiliary_load_kW = 0.0

# --- C. TEPLOTA OLEJE ---
# Teplota oleje, který přitéká do hřídele ke spojce.
T_olej_inlet = 70.0  # [°C]

# --- D. ROZJEZD VE SVAHU (HILL HOLD) ---
# Simulace stání v kopci na prokluzující spojce před rozjezdem.
ENABLE_HILL_START = False 
t_hold       = 1.0     # [s] Jak dlouho stojíme
n_motor_hold = 1800.0  # [rpm] Otáčky při stání

# --- E. PRŮBĚH OTÁČEK (Simulace rozjezdu) ---
# Jak agresivně řidič pouští spojku (1.0 = lineárně).
RPM_SHAPE_FACTOR = 1.0 

# Otáčky motoru během rozjezdu
n_motor_start = 1200.0 # [rpm] Začátek rozjezdu
n_motor_end   = 1200.0 # [rpm] Konec prokluzu
n_motor_idle  = 1200.0 # [rpm] Volnoběh po sepnutí

# Skluz (Rozdíl otáček Motor - Kola)
n_slip_start  = 1200.0 # [rpm] Na začátku kola stojí (rozdíl je plný)
n_slip_end    = 0.0    # [rpm] Na konci je sepnuto (rozdíl je nula)

# --- F. ČASOVÁNÍ CYKLU ---
n_cyklu = 1           # Počet opakování
t_zab   = 3.0         # [s] Doba trvání prokluzu (samotný rozjezd)
t_pauza = 1.0         # [s] Doba chladnutí po rozjezdu

# --- G. GEOMETRIE LAMELY A SEGMENTACE ---
n_segments = 10         # Na kolik kroužků lamelu rozdělíme (pro přesnost)
r_out = 0.124           # [m] Vnější poloměr
r_in = 0.0875           # [m] Vnitřní poloměr
tloustka_oceli = 0.004  # [m] Tloušťka ocelového jádra lamely

# --- H. HYDRAULIKA A DRÁŽKY ---
# Pokud zvolíte režim FIXED, použije se tento průtok:
q_total_lmin_fixed = 5.0  # [L/min]

# Geometrie drážek (Klíčové pro výpočet hydraulického odporu!)
sirka_drazky   = 0.0015   # [m]
hloubka_drazky = 0.0002   # [m]
pocet_drazek   = 40       # [ks] Počet drážek na jedné straně lamely
n_pairs = 14              # Počet třecích ploch (mezer), kudy teče olej

# Logická pojistka (kdyby uživatel zapomněl)
if not ENABLE_HILL_START: t_hold = 0.0

# ==============================================================================
# ČÁST 2: DEFINICE FYZIKÁLNÍCH FUNKCÍ
# ==============================================================================
# Zde definujeme "recepty" (funkce), které budeme v simulaci volat.

def get_steel_props(T_celsius):
    """ 
    Vypočítá vlastnosti oceli podle její aktuální teploty.
    Ocel se chová jinak, když je studená (20°C) a když je žhavá (500°C).
    
    Vstupy:
      T_celsius: Teplota oceli [°C]
      
    Výstupy:
      k:   Tepelná vodivost (jak dobře vede teplo)
      c_p: Tepelná kapacita (kolik tepla do sebe "nasákne")
      rho: Hustota (kolik váží kubík)
    """
    # Oříznutí teploty (aby vzorec nepočítal nesmysly pro extrémní teploty)
    T = np.clip(T_celsius, 20.0, 1000.0)
    
    # Vzorce pro ocel (nelineární závislost)
    k = 54.0 - 0.028 * T
    c_p = 450.0 + 0.28 * T
    rho = 7850.0  # Hustota se mění minimálně, bereme konstantu
    
    return k, c_p, rho

def get_oil_viscosity(T_oil):
    """ 
    Vypočítá viskozitu oleje (jak moc je "hustý" a "lepivý") podle teploty.
    
    Vstupy:
      T_oil: Teplota oleje [°C]
      
    Výstupy:
      nu: Kinematická viskozita [m2/s]
    """
    # Používáme lineární interpolaci mezi dvěma známými body oleje ATF.
    # Při 40°C je hustý (30e-6), při 100°C je řídký (7e-6).
    return np.interp(T_oil, [40, 100], [30e-6, 7e-6])

def get_cooling_from_velocity(v_oil, T_oil_local, Dh):
    """ 
    NOVÁ FUNKCE: Výpočet součinitele přestupu tepla 'h' [W/m2K].
    -----------------------------------------------------------
    Tato funkce už neháda rychlost z otáček kola. 
    Místo toho přijímá přesnou rychlost toku v drážce (v_oil), 
    kterou jsme spočítali z hydraulického tlaku.
    
    Vstupy:
      v_oil:       Rychlost proudění oleje v drážce [m/s] (Vypočteno z tlaku)
      T_oil_local: Místní teplota oleje (pro určení viskozity)
      Dh:          Hydraulický průměr drážky (velikost "trubky")
    """
    # Pokud olej stojí (nebo teče extrémně pomalu), chladí jen minimálně (konvekce)
    if v_oil < 0.001: return 50.0 
    
    # Konstanty pro olej
    rho = 850.0       # Hustota
    lam_oil = 0.14    # Vodivost
    c_oil = 2000.0    # Kapacita
    
    # 1. Zjistíme viskozitu oleje v tomto konkrétním místě
    # (Pokud je olej horký, je řidší -> lépe teče, ale mění se charakter toku)
    nu = get_oil_viscosity(T_oil_local)
    
    # 2. Reynoldsovo číslo (Re)
    # Toto číslo říká, jak moc je tok "divoký" (turbulentní).
    # Re = (Rychlost * Průměr) / Viskozita
    # Čím vyšší rychlost nebo větší trubka, tím vyšší Re.
    # Čím hustší olej (vyšší viskozita), tím nižší Re.
    Re = (v_oil * Dh) / nu
    
    # 3. Prandtl číslo (Pr)
    # Vlastnost kapaliny (poměr viskozity a tepelné vodivosti)
    Pr = (rho * c_oil * nu) / lam_oil
    
    # Bezpečnostní pojistka, aby vzorce nehavarovaly
    if Re < 100: Re = 100 
    
    # 4. Nusseltovo číslo (Nu) - Dittus-Boelterova rovnice
    # Toto číslo říká, jak moc kapalina "krade" teplo stěně.
    # Pro turbulentní proudění platí, že Nu roste s rychlostí (Re^0.8).
    Nu = 0.023 * (Re**0.8) * (Pr**0.3)
    
    # 5. Přepočet na součinitel 'h'
    h_pipe = (Nu / Dh) * lam_oil
    
    # Faktor zvýšení (Enhancement Factor)
    # Drážky nejsou hladká trubka, mají ostré hrany a nátoky.
    # To zvyšuje turbulenci a tím i chlazení. Zvyšujeme výsledek 2x.
    enhancement_factor = 2.0 
    
    return h_pipe * enhancement_factor

def load_engine_map(filename='motor_data.xlsx'):
    """ 
    Načte charakteristiku motoru z Excelu.
    Pokud Excel neexistuje, vytvoří si vymyšlená data (pro ukázku).
    """
    try:
        df = pd.read_excel(filename)
        rpm_data = df['RPM'].values; torque_data = df['Torque'].values
    except FileNotFoundError:
        print(f"INFO: Soubor '{filename}' nenalezen, používám demo data motoru.")
        # Vymyšlená křivka momentu:
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
        
    # Vytvoří funkci, která umí "dopočítat" moment pro jakékoliv otáčky
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data

# ==============================================================================
# ČÁST 3: INICIALIZACE A GEOMETRIE (OPRAVENO - PŘEPÍNAČ ZATÍŽENÍ)
# ==============================================================================

# --- NOVÉ: NASTAVENÍ ROZLOŽENÍ TLAKU ---
# "UNIFORM_WEAR"     = Rovnoměrné opotřebení (Zajetá spojka). Teplo ~ r^2.
#                      Teplo je rozloženo rovnoměrně po ploše.
# "UNIFORM_PRESSURE" = Rovnoměrný tlak (Nová spojka). Teplo ~ r^3.
#                      Vnější okraj se tře rychleji -> Větší teplo na okraji.

MODEL_ZATIZENI = "UNIFORM_WEAR"  # <--- ZDE PŘEPÍNEJTE (WEAR nebo PRESSURE)


# 1. Načtení mapy motoru
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()

# 2. Materiálové vlastnosti
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2

# 3. Výpočet koeficientu BETA
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

# 4. Konstanty oleje
c_oil = 2000.0; lambda_oil = 0.14; rho_oil = 850.0

# --- A. HYDRAULICKÁ GEOMETRIE ---
S_tok_jedna = sirka_drazky * hloubka_drazky
O_tok = 2 * (sirka_drazky + hloubka_drazky)
Dh = 4 * S_tok_jedna / O_tok 

S_flow_total_m2 = n_pairs * pocet_drazek * S_tok_jedna
L_channel = r_out - r_in

S_total_annulus = np.pi * (r_out**2 - r_in**2)
S_grooves_total = pocet_drazek * sirka_drazky * (r_out - r_in)
ratio_groove = S_grooves_total / S_total_annulus

print("-" * 60)
print(f"INFO: GEOMETRIE A MODEL")
print(f"  * Režim chlazení: {CHLAZENI_TYP}")
print(f"  * Model zatížení: {MODEL_ZATIZENI}")
print(f"  * Průtočná plocha: {S_flow_total_m2*1e6:.1f} mm2")
print("-" * 60)


# --- B. GENERACE SEGMENTŮ ---
radii_boundaries = np.linspace(r_in, r_out, n_segments + 1)
segments = [] 
sum_factors = 0.0 

for i in range(n_segments):
    r1 = radii_boundaries[i]
    r2 = radii_boundaries[i+1]
    r_mid = (r1 + r2) / 2
    S_seg = np.pi * (r2**2 - r1**2)
    
    if INCLUDE_AREA_REDUCTION:
        S_heat = S_seg * (1 - ratio_groove) 
    else:
        S_heat = S_seg
    S_cool = S_seg * ratio_groove
    
    # --- ZDE JE TEN PŘEPÍNAČ ---
    if MODEL_ZATIZENI == "UNIFORM_PRESSURE":
        # Tlak je konstantní -> Síla roste s plochou -> Moment roste s ramenem (r)
        # Ve výsledku je teplo úměrné r^3 (rozdíl třetích mocnin)
        # Větší teplo na vnějším okraji.
        factor = (r2**3 - r1**3)
        
    else: # UNIFORM_WEAR (Default)
        # Tlak klesá s poloměrem (p*r = konst) -> Opotřebení je konstantní.
        # Teplo je úměrné ploše (r^2).
        # Teplo je rozloženo "spravedlivě" podle velikosti kroužku.
        factor = (r2**2 - r1**2)
    
    segments.append({
        'id': i, 
        'r_mid': r_mid, 
        'S_seg': S_seg, 
        'S_heat': S_heat, 
        'raw_factor': factor
    })
    sum_factors += factor

# Normalizace
for seg in segments:
    seg['torque_factor'] = seg['raw_factor'] / sum_factors

    # ==============================================================================
# ČÁST 4: PŘÍPRAVA 1D SOLVERU A SMYČKY
# ==============================================================================

# --- A. SÍŤ VE TLOUŠŤCE MATERIÁLU (FDM - Metoda konečných diferencí) ---
# Představte si, že lamelu v řezu rozkrájíme na tenké plátky.
# Počítáme jen polovinu tloušťky, protože lamela je symetrická (teplo jde z obou stran stejně).

L = tloustka_oceli / 2   # [m] Polovina tloušťky
N_nodes = 50             # [ks] Počet uzlů (plátků). Čím víc, tím přesnější, ale pomalejší.
dx = L / (N_nodes - 1)   # [m] Vzdálenost mezi dvěma uzly (tloušťka plátku)

# --- B. MATICE TEPLOT ---
# Vytvoříme tabulku, kde budeme držet aktuální teplotu každého kousku oceli.
# Rozměry: [Počet segmentů (kroužků) x Počet uzlů (hloubka)]
# Na začátku mají všechny body teplotu oleje (70°C).
T_matrix = np.full((n_segments, N_nodes), T_olej_inlet)

# --- C. AUTOMATICKÝ VÝPOČET STABILITY (Časový krok dt) ---
# Aby simulace fungovala, teplo nesmí za jeden krok "přeskočit" celý uzel.
# Musíme najít "nejrychlejší možný přenos tepla" a podle toho nastavit krok.

# 1. Zjistíme vlastnosti studené oceli (za studena vede teplo nejrychleji)
k_fast, c_fast, rho_fast = get_steel_props(20.0) 

# 2. Teplotní difuzivita (Alpha) - Jak rychle se teplo šíří materiálem
alpha_max = k_fast / (rho_fast * c_fast)

# 3. Kritický krok (podle Courantova kritéria stability)
dt_critical = 0.5 * (dx**2) / alpha_max

# 4. Bezpečný krok
# Pro jistotu bereme jen 20 % kritického času.
# (Hydraulika může být na začátku divoká, tak raději zpomalíme).
dt = 0.2 * dt_critical 

print(f"INFO: Automaticky vypočten časový krok dt = {dt:.6f} s")

# --- D. PŘÍPRAVA LOGOVÁNÍ (Místo pro ukládání výsledků) ---
# Spočítáme celkový čas simulace
t_cyklus = t_hold + t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Prázdné seznamy, do kterých budeme ukládat data pro grafy
cas_plot = []

# Teploty (budeme sledovat jen vybraná místa)
T_surf_mid_log = []   # Povrchová teplota uprostřed lamely
T_core_mid_log = []   # Teplota jádra uprostřed lamely

# Výkonová data
torque_log = []       # Moment [Nm]
power_log = []        # Výkon [W]
rpm_log = []          # Otáčky [rpm]

# Data o chlazení (h)
h_inner_log = []      # h na vnitřním okraji
h_mid_log = []        # h uprostřed
h_outer_log = []      # h na vnějším okraji

# Data o rozložení teploty (pro kontrolu Hotspotů)
T_surf_R_in_log = []  # Povrch vnitřního okraje
T_surf_R_out_log = [] # Povrch vnějšího okraje
oil_temp_out_log = [] # Teplota oleje, který vytéká ven

# Indexy pro snadný přístup k vnitřnímu, střednímu a vnějšímu segmentu
idx_mid = n_segments // 2
idx_out = n_segments - 1

# --- E. AKUMULÁTORY ENERGIE ---
# Proměnné pro závěrečnou kontrolu (Validaci)
E_oil_removed_cum_J = 0.0  # Kolik energie celkem odnesl olej [Joule]

# Proměnné pro hledání maximálních hodnot (špičky)
max_torque_rec = 0.0
max_power_net_rec = 0.0
max_q_net_rec = 0.0

# --- F. START SIMULACE ---
t = 0.0    # Čas nula
step = 0   # Krok nula

# ------------------------------------------------------------------------------
# 5. HLAVNÍ SIMULAČNÍ SMYČKA (OPRAVENO - LOGOVÁNÍ)
# ------------------------------------------------------------------------------
print("... Spouštím simulaci ...")

while t < t_total:
    
    t_local = t % t_cyklus
    
    # --- A. KINEMATIKA ---
    if t_local < t_hold:
        rpm_engine = n_motor_hold; rpm_slip = n_motor_hold
    elif t_local < (t_hold + t_zab):
        t_in = t_local - t_hold
        ratio = (t_in / t_zab) ** RPM_SHAPE_FACTOR
        rpm_engine = n_motor_start + (n_motor_end - n_motor_start) * ratio
        rpm_slip = n_slip_start * (1 - ratio)
    else:
        rpm_engine = n_motor_idle; rpm_slip = 0.0

    torque_total = get_torque_from_rpm(rpm_engine)
    
    # --- B. VÝKON ---
    omega_slip = rpm_slip * 2 * np.pi / 60
    P_net_total = max(0.0, (torque_total * omega_slip) - (P_auxiliary_load_kW * 1000))

    # --- C. HYDRAULIKA ---
    v_oil_channel = 0.0       
    mdot_total_kg_s = 0.0     
    
    if CHLAZENI_TYP in ["ANALYTIC_RPM", "ANALYTIC_FIX_TEMPERATURE"]:
        omega_engine = rpm_engine * 2 * np.pi / 60
        dP_centrifugal = 0.5 * rho_oil * (omega_engine**2) * (r_out**2 - r_in**2)
        mu_dynamic = get_oil_viscosity(T_olej_inlet) * rho_oil
        K_shape = 24.0
        
        if dP_centrifugal > 0:
            v_oil_channel = (dP_centrifugal * Dh**2) / (K_shape * mu_dynamic * L_channel)
        else:
            v_oil_channel = 0.0
            
        Q_vol_m3_s = v_oil_channel * S_flow_total_m2
        mdot_total_kg_s = Q_vol_m3_s * rho_oil
        
    elif CHLAZENI_TYP == "ANALYTIC_FIXED":
        mdot_total_kg_s = (q_total_lmin_fixed / 60 / 1000) * rho_oil
        Q_vol_m3_s = mdot_total_kg_s / rho_oil
        v_oil_channel = Q_vol_m3_s / S_flow_total_m2
        
    elif CHLAZENI_TYP == "NO_COOLING":
        v_oil_channel = 0.0
        mdot_total_kg_s = 0.0

    mdot_per_gap = mdot_total_kg_s / n_pairs

    # --- D. SMYČKA PŘES SEGMENTY ---
    T_oil_current = T_olej_inlet 
    
    # Pomocné proměnné pro logování
    h_curr_in = 0
    h_curr_mid = 0 # <--- OPRAVA: Inicializace
    h_curr_out = 0
    
    P_removed_oil_step_W = 0.0
    max_q_step = 0.0
    
    for i in range(n_segments):
        seg = segments[i]
        
        # 1. Zdroj
        P_seg = P_net_total * seg['torque_factor']
        q_gen = (P_seg / (n_pairs * seg['S_heat'])) * beta
        
        # 2. Teplota oleje pro výpočet h
        if CHLAZENI_TYP == "ANALYTIC_FIX_TEMPERATURE":
            T_oil_calc = T_olej_inlet
        else:
            T_oil_calc = T_oil_current

        # 3. Výpočet h
        if CHLAZENI_TYP == "NO_COOLING":
            h_seg = 0.0
        else:
            h_seg = get_cooling_from_velocity(v_oil_channel, T_oil_calc, Dh)
            
        # ZACHYCENÍ HODNOT PRO LOGOVÁNÍ
        if i == 0: h_curr_in = h_seg
        if i == idx_mid: h_curr_mid = h_seg # <--- OPRAVA: Zde se ukládá střední hodnota
        if i == idx_out: h_curr_out = h_seg
        
        # 4. Bilance
        T_surf = T_matrix[i, 0]
        q_cool = h_seg * (T_surf - T_oil_calc)
        q_net = q_gen - q_cool
        
        if q_net > max_q_step: max_q_step = q_net
        
        # 5. Ohřev oleje
        Q_absorbed = q_cool * seg['S_heat']
        
        if mdot_per_gap > 1e-9:
            dT_oil = Q_absorbed / (mdot_per_gap * c_oil)
        else:
            dT_oil = 100.0 
            
        T_oil_current += dT_oil
        P_removed_oil_step_W += Q_absorbed
        
        # 6. FDM Solver
        k_vec, cp_vec, rho_val = get_steel_props(T_matrix[i, :])
        alpha_vec = k_vec / (rho_val * cp_vec)
        
        T_old = T_matrix[i, :]
        T_new_seg = np.copy(T_old)
        
        T_new_seg[1:-1] = T_old[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T_old[2:] - 2*T_old[1:-1] + T_old[:-2])
        T_new_seg[0] = T_old[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T_old[0] - T_old[1]) / dx)
        T_new_seg[-1] = T_old[-1] + alpha_vec[-1] * dt / dx**2 * (T_old[-2] - T_old[-1])
        
        T_matrix[i, :] = T_new_seg[:]

    # --- E. UPDATE DAT ---
    E_oil_removed_cum_J += P_removed_oil_step_W * n_pairs * dt
    
    if torque_total > max_torque_rec: max_torque_rec = torque_total
    if P_net_total > max_power_net_rec: max_power_net_rec = P_net_total
    if max_q_step > max_q_net_rec: max_q_net_rec = max_q_step
    
    t += dt
    step += 1
    
    if step % 500 == 0:
        cas_plot.append(t)
        T_surf_mid_log.append(T_matrix[idx_mid, 0])
        T_core_mid_log.append(T_matrix[idx_mid, -1])
        torque_log.append(torque_total)
        power_log.append(P_net_total)
        rpm_log.append(rpm_engine)
        
        # LOGOVÁNÍ HODNOT H
        h_inner_log.append(h_curr_in)
        h_mid_log.append(h_curr_mid)   # <--- OPRAVA: Zde chybělo přidání do seznamu
        h_outer_log.append(h_curr_out)
        
        T_surf_R_in_log.append(T_matrix[0, 0])
        T_surf_R_out_log.append(T_matrix[idx_out, 0])
        oil_temp_out_log.append(T_oil_current)
#------------------------------------
# 1. VÝPOČET ENERGETICKÉ BILANCE (Kontrola fyziky)
# ------------------------------------------------------------------------------
# Energie se nesmí ztratit. Musí platit:
# VSTUP (Motor) = VÝSTUP (Teplo v oceli + Teplo v oleji)

# A. Vstupní energie
# Sečteme výkon v každém čase (Integrál výkonu podle času)
try:
    # Novější verze NumPy
    E_input_J = np.trapezoid(power_log, cas_plot)
except AttributeError:
    # Starší verze NumPy
    E_input_J = np.trapz(power_log, cas_plot)

# Do spojky jde jen část energie (Beta), zbytek jde do papírového obložení.
E_target_J = E_input_J * beta

# B. Energie uložená v oceli (Tepelná setrvačnost)
# Musíme projít každý malý kousek (uzel) oceli a zjistit, o kolik se ohřál.
E_stored_global_steel_J = 0.0

for i in range(n_segments):
    seg = segments[i]
    energ_segmentu = 0.0
    for j in range(N_nodes):
        T_node_end = T_matrix[i, j]
        
        # Průměrná tepelná kapacita pro daný rozsah teplot
        T_avg = (T_node_end + T_olej_inlet) / 2
        _, cp_avg, _ = get_steel_props(T_avg)
        
        # Hmotnost uzlu
        vol_node = seg['S_seg'] * dx 
        m_node = vol_node * rho_s_ref
        
        # Energie = Hmotnost * Kapacita * (Konečná teplota - Počáteční teplota)
        energ_segmentu += m_node * cp_avg * (T_node_end - T_olej_inlet)
        
    E_stored_global_steel_J += energ_segmentu

# Vynásobíme počtem třecích ploch (protože počítáme jednu lamelu, ale je jich tam víc)
E_stored_global_steel_J *= n_pairs

# C. Energie odnesená olejem (Chlazení)
# Toto jsme sčítali průběžně v hlavní smyčce.
E_cooled_J = E_oil_removed_cum_J

# D. Celkový součet a Chyba
E_accounted_J = E_stored_global_steel_J + E_cooled_J
diff = E_target_J - E_accounted_J

err_percent = 0.0
if E_target_J > 0:
    err_percent = abs(diff / E_target_J) * 100

# Pokud je chyba pod 5 %, model považujeme za platný.
validation_status = "PASS" if err_percent < 5.0 else "WARNING"

# ------------------------------------------------------------------------------
# 2. VÝPIS VÝSLEDKŮ DO KONZOLE
# ------------------------------------------------------------------------------

print(f"1. VSTUP: Energie od motoru (korekovaná o Betu): {E_target_J/1000:.1f} kJ")
print("-" * 40)
print(f"2. NALEZENO: Teplo akumulované v oceli:          {E_stored_global_steel_J/1000:.1f} kJ")
print(f"3. ODVEDENO: Teplo odnesené olejem:              {E_cooled_J/1000:.1f} kJ")
print(f"   SOUČET (Nalezeno + Odvedeno):                 {E_accounted_J/1000:.1f} kJ")
print("-" * 40)
print(f"ROZDÍL (Chyba modelu):                           {diff/1000:.1f} kJ")
print(f"PROCENTUÁLNÍ CHYBA:                              {err_percent:.2f} %")
print(f"-> VERDIKT: {validation_status}")

print("\n" + "="*60)
print(" SOUHRNNÁ STATISTIKA")
print("="*60)

# Zjištění maximální teploty oleje (pokud nějaký tekl)
max_oil = max(oil_temp_out_log) if len(oil_temp_out_log) > 0 else T_olej_inlet

print(f"Maximální teplota povrchu oceli:  {max(T_surf_R_out_log):.1f} °C")
print(f"Maximální teplota výstupního oleje: {max_oil:.1f} °C")
print(f"Špičkový tepelný tok:             {max_q_net_rec / 1e6:.2f} MW/m²")

if CHLAZENI_TYP == "ANALYTIC_RPM":
    print("Režim: SAMONASÁVÁNÍ (REALITA - Olej se ohřívá)")
elif CHLAZENI_TYP == "ANALYTIC_FIX_TEMPERATURE":
    print("Režim: SAMONASÁVÁNÍ (TEORIE - Olej se neohřívá)")
elif CHLAZENI_TYP == "ANALYTIC_FIXED":
    print(f"Režim: FIXNÍ PRŮTOK ({q_total_lmin_fixed} L/min)")

# ------------------------------------------------------------------------------
# 3. VYKRESLENÍ GRAFŮ
# ------------------------------------------------------------------------------
# Vytvoříme plátno se 4 grafy pod sebou

fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

# --- GRAF 1: Teplotní průběh uprostřed lamely ---
axs[0].plot(cas_plot, T_surf_mid_log, 'r-', label='Povrch Oceli (Střední R)')
axs[0].plot(cas_plot, T_core_mid_log, 'b--', label='Jádro Oceli (Střední R)')
axs[0].set_ylabel('Teplota [°C]', fontweight='bold')
axs[0].set_title(f'1. Teplotní gradient v materiálu (Poloměr = {segments[idx_mid]["r_mid"]*1000:.1f} mm)', fontsize=12)
axs[0].grid(True, alpha=0.5)
axs[0].legend()

# --- GRAF 2: Zátěž (Moment a Výkon) ---
axs[1].set_ylabel('Moment [Nm]', color='green', fontweight='bold')
axs[1].plot(cas_plot, torque_log, 'g-', label='Moment motoru')
axs[1].tick_params(axis='y', labelcolor='green')
axs[1].grid(True, alpha=0.5)

# Přidáme druhou osu Y pro výkon
ax2 = axs[1].twinx()
ax2.set_ylabel('Výkon [kW]', color='orange', fontweight='bold')
ax2.plot(cas_plot, [p/1000 for p in power_log], 'orange', linestyle='--', label='Čistý výkon')
ax2.tick_params(axis='y', labelcolor='orange')
axs[1].set_title('2. Zatížení spojky (Zdroj tepla)', fontsize=12)

# --- GRAF 3: Chlazení (h) ---
axs[2].plot(cas_plot, h_inner_log, 'b:', label='h - Vnitřní okraj')
axs[2].plot(cas_plot, h_mid_log, 'b-', label='h - Střed')
axs[2].plot(cas_plot, h_outer_log, 'b--', label='h - Vnější okraj')
axs[2].set_ylabel('h [W/m2K]', color='blue', fontweight='bold')
axs[2].set_title(f'3. Součinitel h (Hydraulika: Tlakový rozdíl)', fontsize=12)
axs[2].grid(True, alpha=0.5)
axs[2].legend()

# --- GRAF 4: Rozložení teploty (Vnitřek vs Vnějšek) ---
# Tady uvidíte, jestli se pálí okraje (Hotspoty)
axs[3].plot(cas_plot, T_surf_R_in_log, 'g:', label='Vnitřní poloměr (Vstup oleje)')
axs[3].plot(cas_plot, T_surf_R_out_log, 'purple', linestyle='--', label='Vnější poloměr (Výstup oleje)')
axs[3].set_ylabel('Teplota [°C]', fontweight='bold')
axs[3].set_xlabel('Čas simulace [s]', fontweight='bold')
axs[3].set_title('4. Rozdíl teplot na okrajích lamely', fontsize=12)
axs[3].grid(True, alpha=0.5)
axs[3].legend()

# Přidáme textové okno s výsledkem validace do posledního grafu
valid_color = 'green' if validation_status == "PASS" else 'red'
info_text = (f"BILANCE: {validation_status}\nChyba: {err_percent:.2f}%")
axs[3].text(0.02, 0.95, info_text, transform=axs[3].transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=valid_color))

plt.tight_layout()
plt.show()
