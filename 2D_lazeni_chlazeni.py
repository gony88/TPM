# ==============================================================================
# ČÁST 1: IMPORTY A UŽIVATELSKÉ NASTAVENÍ (KONFIGURACE)
# ==============================================================================

import numpy as np                  # Jádro výpočtů (matice, pole)
import matplotlib.pyplot as plt     # Vykreslování grafů
import pandas as pd                 # Práce s daty (Excel)
from scipy.interpolate import interp1d # Interpolace map motoru

# ==============================================================================
#                       1. UŽIVATELSKÉ NASTAVENÍ
# ==============================================================================

# --- A. NASTAVENÍ REŽIMU CHLAZENÍ ---
# Vyberte jeden z následujících režimů:
#
# 1. "ANALYTIC_FIXED" 
#    - Průtok je KONSTANTNÍ (zadáno v L/min).
#    - Olej se průchodem drážkami OHŘÍVÁ (Sériový tok: Segment 0 -> Segment 9).
#    - H se mění dynamicky dle otáček a lokální viskozity.
#
# 2. "ANALYTIC_RPM" (Samonasávání)
#    - Průtok KOLÍSÁ podle otáček motoru (Odstředivé čerpadlo).
#    - Olej se průchodem OHŘÍVÁ.
#    - Nejpřesnější simulace reálného auta bez externího čerpadla.
#
# 3. "ANALYTIC_FIX_TEMPERATURE" (Pro porovnání s 1D)
#    - Průtok je fixní (použito pro výpočet h), ale...
#    - Olej se NEOHŘÍVÁ! V každém bodě lamely má vstupní teplotu.
#    - Slouží k validaci oproti jednoduchým 1D modelům.
#
# 4. "NO_COOLING"
#    - Žádné chlazení (Adiabatický děj).

CHLAZENI_TYP = "NO_COOLING"  # <--- ZDE ZVOLTE REŽIM

# Započítat zmenšení plochy o drážky? (True = menší plocha = větší teplo)
INCLUDE_AREA_REDUCTION = False

# --- B. ODBĚR VÝKONU (PT0 / Hydraulika) ---
# Kolik kW sebere čerpadlo nástavby, než se výkon dostane ke spojce.
P_auxiliary_load_kW = 0.0

# --- C. TEPLOTA OLEJE ---
# Teplota oleje na vstupu do hřídele.
T_olej_inlet = 70.0  # [°C]

# --- D. ROZJEZD VE SVAHU ---
ENABLE_HILL_START = False 
t_hold       = 1.0     # [s] Doba držení
n_motor_hold = 1800.0  # [rpm]

# --- E. PRŮBĚH OTÁČEK (Simulace řidiče) ---
RPM_SHAPE_FACTOR = 1.0 

n_motor_start = 1200.0 # [rpm] Start (Volnoběh)
n_motor_end   = 1200.0 # [rpm] Konec (Vytočený motor)
n_motor_idle  = 1200.0 # [rpm] Volnoběh po sepnutí

n_slip_start  = 1200.0 # [rpm]
n_slip_end    = 0.0    # [rpm]

# --- F. ČASOVÁNÍ CYKLU ---
n_cyklu = 1          
t_zab   = 3.0        # [s] Doba prokluzu
t_pauza = 1.0        # [s]

# --- G. GEOMETRIE A SEGMENTACE ---
n_segments = 10         
r_out = 0.124           # [m]
r_in = 0.0875           # [m]
tloustka_oceli = 0.006  # [m] (Tenká lamela se rychleji prohřeje)

# --- H. HYDRAULIKA A DRÁŽKY ---
# Zde jsou parametry pro všechny typy chlazení.

# 1. Pro režimy FIXED a FIX_TEMPERATURE
q_total_lmin_fixed = 5.0  # [L/min] 

# 2. Pro režim RPM (Samonasávání) - Geometrie drážek
sirka_drazky   = 0.0015   # [m]
hloubka_drazky = 0.0002   # [m]
pocet_drazek   = 40       # [ks] Celkový počet drážek na jedné lamele

# Počet třecích mezer (mezi kolik ploch se olej dělí)
n_pairs = 14            

# Logická pojistka
if not ENABLE_HILL_START: t_hold = 0.0

# ==============================================================================
# ČÁST 2: DEFINICE FYZIKÁLNÍCH FUNKCÍ
# ==============================================================================

def get_steel_props(T_celsius):
    """ 
    Vlastnosti oceli v závislosti na teplotě (C_p, k). 
    Zahrnuje nelinearitu materiálu.
    """
    T = np.clip(T_celsius, 20.0, 1000.0)
    k = 54.0 - 0.028 * T
    c_p = 450.0 + 0.28 * T
    rho = 7850.0
    return k, c_p, rho

def get_oil_viscosity(T_oil):
    """ 
    Kinematická viskozita oleje [m2/s] dle teploty. 
    Klíčové pro výpočet Reynoldsova čísla.
    """
    return np.interp(T_oil, [40, 100], [30e-6, 7e-6])

def get_cooling_analytical_local(rpm, T_oil_local, r_segment_mid, Dh):
    """ 
    Výpočet součinitele přestupu tepla 'h' [W/m2K].
    Toto je jádro analytického modelu.
    """
    if rpm < 10: return 50.0 # Minimální konvekce při stání
    
    rho = 850.0; lam_oil = 0.14; c_oil = 2000.0
    
    # 1. Zjistíme viskozitu pro aktuální teplotu
    nu = get_oil_viscosity(T_oil_local)
    
    # 2. Rychlost toku v drážce (odstředivá síla)
    omega = rpm * (2 * np.pi / 60)
    v_oil = omega * r_segment_mid
    
    # 3. Bezrozměrná čísla (Re, Pr)
    Re = (v_oil * Dh) / nu
    Pr = (rho * c_oil * nu) / lam_oil
    
    if Re < 100: Re = 100 # Pojistka proti dělení nulou
    
    # 4. Nusseltovo číslo (Turbulentní korelace Dittus-Boelter)
    # H roste s otáčkami (Re^0.8)
    Nu = 0.023 * (Re**0.8) * (Pr**0.3)
    
    # 5. Výsledné h
    h_pipe = (Nu / Dh) * lam_oil
    enhancement_factor = 2.0 # Korekce na drsnost drážek a nátokové hrany
    return h_pipe * enhancement_factor

def load_engine_map(filename='motor_data.xlsx'):
    """ Načtení mapy motoru z Excelu. """
    try:
        df = pd.read_excel(filename)
        rpm_data = df['RPM'].values; torque_data = df['Torque'].values
    except FileNotFoundError:
        print(f"INFO: Soubor nenalezen, používám demo data motoru.")
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data
# ==============================================================================
# ČÁST 3: INICIALIZACE A GEOMETRIE
# ==============================================================================

# Načtení mapy a konstant
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)

# Vlastnosti třecího materiálu (papír/karbon)
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2

# Koeficient BETA (Rozdělení tepla mezi ocel a obložení)
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

# Konstanty oleje (pro výpočty mimo funkce)
c_oil = 2000.0; lambda_oil = 0.14; rho_oil = 850.0

# --- A. VÝPOČET GEOMETRIE DRÁŽEK (PRO SAMONASÁVÁNÍ) ---
# Potřebujeme znát celkovou "díru", kudy olej teče ven.

# 1. Průřez jedné drážky [m2]
S_tok_jedna = sirka_drazky * hloubka_drazky

# 2. Hydraulický průměr (Dh) - důležité pro Reynoldsovo číslo
O_tok = 2 * (sirka_drazky + hloubka_drazky)
Dh = 4 * S_tok_jedna / O_tok 

# 3. Celková průtočná plocha celé spojky (všechny lamely paralelně)
# Olej teče skrze 'n_pairs' mezer a v každé mezeře je 'pocet_drazek'.
S_flow_total_m2 = n_pairs * pocet_drazek * S_tok_jedna

# 4. Výpočet poměru plochy drážek (ratio_groove)
# Kolik % plochy lamely tvoří drážky? (Důležité pro redukci plochy)
# Plocha mezikruží:
S_total_annulus = np.pi * (r_out**2 - r_in**2)
# Plocha všech drážek na jedné lamele (zjednodušeně obdélníky):
S_grooves_total = pocet_drazek * sirka_drazky * (r_out - r_in)
# Výsledný poměr:
ratio_groove = S_grooves_total / S_total_annulus

print("-" * 60)
print(f"INFO: GEOMETRIE A CHLAZENÍ")
print(f"  * Režim: {CHLAZENI_TYP}")
print(f"  * Průřez drážek (celkový):  {S_flow_total_m2*1e6:.1f} mm2")
print(f"  * Poměr plochy drážek:      {ratio_groove*100:.1f} %")
print("-" * 60)


# --- B. GENERACE SEGMENTŮ A ROZDĚLENÍ MOMENTU ---
radii_boundaries = np.linspace(r_in, r_out, n_segments + 1)
segments = [] 

# MODEL ZATÍŽENÍ:
# Zde volíme, zda je spojka "Nová" (Uniform Pressure) nebo "Zajetá" (Uniform Wear).
# Pro bezpečnostní výpočty se obvykle používá UNIFORM PRESSURE (r^3), 
# protože vytváří větší teplo na vnějším okraji (Hotspot).

# Pro Uniform Pressure (r^3):
denom_torque = (r_out**2 - r_in**2) # (Tato normalizace je složitější, zjednodušeno pro poměr)
# Ve skutečnosti pro r^3 platí, že teplo roste lineárně s r.
# Pro Uniform Wear (r^2) je teplo konstantní.

# Zde použijeme logiku rozdělení výkonu podle poloměru:
# Faktor ~ r * Plocha ~ r * r = r^2 (Uniform Wear) 
# Faktor ~ r * r * r (Uniform Pressure - Vnější okraj bere víc)

# Pro tento kód použijeme UNIFORM PRESSURE (Vnější okraj topí víc), 
# aby byl vidět rozdíl v 2D modelu.
sum_factors = 0.0
temp_segments = []

for i in range(n_segments):
    r1 = radii_boundaries[i]
    r2 = radii_boundaries[i+1]
    r_mid = (r1 + r2) / 2
    S_seg = np.pi * (r2**2 - r1**2)
    
    # Rozdělení ploch
    if INCLUDE_AREA_REDUCTION:
        S_heat = S_seg * (1 - ratio_groove) 
    else:
        S_heat = S_seg
    S_cool = S_seg * ratio_groove
    
 # MODEL ZATÍŽENÍ:
    
    # 1. UNIFORM WEAR (Rovnoměrné opotřebení - r^2)
    # ZDE MUSÍ BÝT KŘIVKY V REŽIMU NO_COOLING PŘEKRYTÉ
    # Faktor je úměrný ploše segmentu.
    factor_uniform_wear = (r2**2 - r1**2)

    # 2. UNIFORM PRESSURE (Rovnoměrný tlak - r^3)
    # ZDE BUDE VNĚJŠEK TEPLEJŠÍ (HOTSPOT)
    # Faktor zohledňuje plochu * rychlost (rameno).
    factor_uniform_pressure = (r2**3 - r1**3)
    
    # --- VOLBA MODELU ---
    # Odkomentujte ten, který chcete použít:
    
    factor = factor_uniform_wear      # <--- VOLBA: PŘEKRYTÍ KŘIVEK
    # factor = factor_uniform_pressure  # <--- VOLBA: ROZDÍLNÉ KŘIVKY
    
    temp_segments.append({
        'id': i, 'r_mid': r_mid, 'S_seg': S_seg, 'S_heat': S_heat, 'S_cool': S_cool,
        'r1': r1, 'r2': r2, 'raw_factor': factor
    })
    sum_factors += factor

# Normalizace faktorů (aby součet byl 1.0)
for seg in temp_segments:
    seg['torque_factor'] = seg['raw_factor'] / sum_factors
    segments.append(seg)


# ==============================================================================
# ČÁST 4: PŘÍPRAVA 1D SOLVERU A SMYČKY
# ==============================================================================

# Síť v tloušťce materiálu
L = tloustka_oceli / 2   # Symetrie
N_nodes = 50             # Počet uzlů
dx = L / (N_nodes - 1)   # Krok sítě [m]

# Matice teplot [Segmenty x Uzly]
# Na začátku mají všechny body teplotu oleje
T_matrix = np.full((n_segments, N_nodes), T_olej_inlet)

# --- AUTOMATICKÝ VÝPOČET STABILITY (dt) ---
# 1. Nejrychlejší difuze (studená ocel)
k_fast, c_fast, rho_fast = get_steel_props(20.0) 
alpha_max = k_fast / (rho_fast * c_fast)

# 2. Kritický krok
dt_critical = 0.5 * (dx**2) / alpha_max

# 3. Bezpečný krok (Safe Mode)
# Pokud používáme samonasávání, průtok může být na začátku mizivý.
# Olej se bude hřát extrémně rychle. Musíme zpomalit simulaci.
dt = 0.2 * dt_critical 

print(f"INFO: Automaticky vypočten dt = {dt:.6f} s")

# --- PŘÍPRAVA LOGOVÁNÍ ---
t_cyklus = t_hold + t_zab + t_pauza
t_total = n_cyklu * t_cyklus

cas_plot = []
T_surf_mid_log = []; T_core_mid_log = []
torque_log = []; power_log = []; rpm_log = []
h_inner_log = []; h_mid_log = []; h_outer_log = []
T_surf_R_in_log = []; T_surf_R_mid_log = []; T_surf_R_out_log = []
oil_temp_out_log = [] # Log výstupní teploty oleje

idx_mid = n_segments // 2
idx_out = n_segments - 1

# Akumulátory
E_oil_removed_cum_J = 0.0 
max_torque_rec = 0.0; max_power_net_rec = 0.0; max_q_net_rec = 0.0

# Inicializace
t = 0.0
step = 0

print("... Spouštím simulaci ...")

# ------------------------------------------------------------------------------
# 5. HLAVNÍ SIMULAČNÍ SMYČKA
# ------------------------------------------------------------------------------
while t < t_total:
    
    t_local = t % t_cyklus
    
    # --- A. KINEMATIKA (Otáčky a Moment) ---
    if t_local < t_hold:
        # Hill Hold
        rpm_engine = n_motor_hold
        rpm_slip = n_motor_hold
        torque_total = get_torque_from_rpm(rpm_engine)
        
    elif t_local < (t_hold + t_zab):
        # Rozjezd
        t_in = t_local - t_hold
        ratio_lin = t_in / t_zab
        ratio = ratio_lin ** RPM_SHAPE_FACTOR
        
        rpm_engine = n_motor_start + (n_motor_end - n_motor_start) * ratio
        rpm_slip = n_slip_start * (1 - ratio)
        torque_total = get_torque_from_rpm(rpm_engine)
        
    else:
        # Pauza
        rpm_engine = n_motor_idle
        rpm_slip = 0.0
        torque_total = 0.0

    # --- B. VÝKON ---
    omega_slip = rpm_slip * 2 * np.pi / 60
    P_gross = torque_total * omega_slip
    P_net_total = max(0.0, P_gross - (P_auxiliary_load_kW * 1000))

    # --- C. HYDRAULIKA (VÝPOČET PRŮTOKU - PŘEPÍNAČ REŽIMŮ) ---
    
    if CHLAZENI_TYP == "ANALYTIC_RPM":
        # SAMONASÁVÁNÍ: Průtok závisí na otáčkách motoru
        pumping_eff = 0.35 
        omega_engine = rpm_engine * 2 * np.pi / 60
        
        # Radiální rychlost na výstupu (odstředivka)
        v_rad_oil = pumping_eff * omega_engine * r_out
        
        # Celkový průtok = Plocha * Rychlost
        Q_vol_m3_s = S_flow_total_m2 * v_rad_oil
        
        # Pojistka proti nule (aby nevznikla singularita v ohřevu)
        min_flow = (0.1 / 60 / 1000) 
        if Q_vol_m3_s < min_flow: Q_vol_m3_s = min_flow
        
        mdot_total_kg_s = Q_vol_m3_s * rho_oil
        
    elif CHLAZENI_TYP == "NO_COOLING":
        mdot_total_kg_s = 1.0 # Fiktivní hodnota (h bude 0, takže na tom nezáleží)
        
    else: 
        # ANALYTIC_FIXED nebo ANALYTIC_FIX_TEMPERATURE
        # Průtok je dán uživatelským nastavením
        mdot_total_kg_s = (q_total_lmin_fixed / 60 / 1000) * rho_oil

    # Rozdělení průtoku na jednu lamelu
    mdot_per_surface = mdot_total_kg_s / n_pairs
    # --- D. SMYČKA PŘES SEGMENTY (TEPLO A CHLAZENÍ) ---
    # Na začátku každého časového kroku přitéká k vnitřnímu okraji čerstvý olej
    T_oil_current = T_olej_inlet 
    
    # Proměnné pro logování v tomto kroku (uložíme si hodnoty z klíčových míst)
    h_curr_in = 0; h_curr_mid = 0; h_curr_out = 0
    max_q_step = 0.0
    P_removed_oil_step_W = 0.0 # Kolik tepla olej odnesl v tomto kroku celkem
    
    for i in range(n_segments):
        seg = segments[i]
        
        # 1. Generace tepla (Zdroj)
        # Rozdělíme celkový výkon podle "Torque Factoru" (vypočteno v Části 2)
        P_seg = P_net_total * seg['torque_factor']
        
        # Přepočet na tok na m2 (zohledňuje Betu)
        q_gen = (P_seg / (n_pairs * seg['S_heat'])) * beta
        
        # 2. Určení teploty oleje pro výpočet chlazení
        # Zde je rozdíl mezi režimy!
        if CHLAZENI_TYP == "ANALYTIC_FIX_TEMPERATURE":
            # Režim pro validaci s 1D: Olej chladí, jako by měl stále vstupní teplotu.
            # (Ignorujeme, že se ohřál v předchozím segmentu)
            T_oil_for_calc = T_olej_inlet
        else:
            # Reálný režim (RPM, FIXED): Olej je teplejší o to, co nabral cestou.
            T_oil_for_calc = T_oil_current

        # 3. Výpočet h (Přestup tepla)
        if CHLAZENI_TYP == "NO_COOLING":
            h_seg = 0.0
        else:
            # Pro všechny analytické režimy (FIXED, RPM, FIX_TEMP) se h počítá stejně
            # (dynamicky dle otáček a lokální teploty oleje)
            h_seg = get_cooling_analytical_local(rpm_engine, T_oil_for_calc, seg['r_mid'], Dh)
            
        # Uložení pro grafy (jen vybrané body)
        if i == 0: h_curr_in = h_seg
        if i == idx_mid: h_curr_mid = h_seg
        if i == idx_out: h_curr_out = h_seg
        
        # 4. Bilance povrchu (Newtonův zákon)
        T_surf = T_matrix[i, 0]
        # q_cool = h * (T_povrch - T_olej)
        q_cool = h_seg * (T_surf - T_oil_for_calc)
        
        # Výsledný tok do materiálu
        q_net = q_gen - q_cool
        
        if q_net > max_q_step: max_q_step = q_net
        
        # 5. Ohřev oleje (Hydraulická vazba)
        # Kolik tepla jsme oleji předali?
        Q_absorbed = q_cool * seg['S_heat']
        
        # O kolik se olej ohřeje, než doteče do dalšího segmentu?
        # dT = Q / (m * c)
        if mdot_per_surface > 1e-9: # Ošetření dělení nulou
            dT_oil = Q_absorbed / (mdot_per_surface * c_oil)
        else:
            dT_oil = 100.0 # Pokud neteče nic, teplota letí nahoru (limitováno)

        # Aktualizace teploty oleje pro DALŠÍ segment
        T_oil_current += dT_oil 
        
        # Sčítání odvedené energie (pro validaci)
        P_removed_oil_step_W += Q_absorbed
        
        # 6. FDM Solver (Vedení tepla dovnitř oceli)
        # Získání vlastností pro aktuální teplotu
        k_vec, cp_vec, rho_val = get_steel_props(T_matrix[i, :])
        alpha_vec = k_vec / (rho_val * cp_vec)
        
        T_old = T_matrix[i, :]
        T_new_seg = np.copy(T_old)
        
        # A) Vnitřní uzly
        T_new_seg[1:-1] = T_old[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T_old[2:] - 2*T_old[1:-1] + T_old[:-2])
        
        # B) Povrchový uzel (Okrajová podmínka q_net)
        T_new_seg[0] = T_old[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T_old[0] - T_old[1]) / dx)
        
        # C) Středový uzel (Symetrie/Izolace)
        T_new_seg[-1] = T_old[-1] + alpha_vec[-1] * dt / dx**2 * (T_old[-2] - T_old[-1])
        
        # Uložení
        T_matrix[i, :] = T_new_seg[:]

    # --- E. UPDATE GLOBÁLNÍCH DAT ---
    # Energie [J] = Výkon [W] * čas [s] * počet lamel
    E_oil_removed_cum_J += P_removed_oil_step_W * n_pairs * dt
    
    if torque_total > max_torque_rec: max_torque_rec = torque_total
    if P_net_total > max_power_net_rec: max_power_net_rec = P_net_total
    if max_q_step > max_q_net_rec: max_q_net_rec = max_q_step
    
    t += dt
    step += 1
    
    # Logování (zředění dat - ukládáme každý 500. krok)
    if step % 500 == 0:
        cas_plot.append(t)
        T_surf_mid_log.append(T_matrix[idx_mid, 0])
        T_core_mid_log.append(T_matrix[idx_mid, -1])
        torque_log.append(torque_total)
        power_log.append(P_net_total)
        rpm_log.append(rpm_engine)
        h_inner_log.append(h_curr_in)
        h_mid_log.append(h_curr_mid)
        h_outer_log.append(h_curr_out)
        T_surf_R_in_log.append(T_matrix[0, 0])
        T_surf_R_mid_log.append(T_matrix[idx_mid, 0])
        T_surf_R_out_log.append(T_matrix[idx_out, 0])
        oil_temp_out_log.append(T_oil_current) # Výstupní teplota z posledního segmentu

# ==============================================================================
# ČÁST 6: VYHODNOCENÍ, VALIDACE A VYKRESLENÍ
# ==============================================================================

print("\n" + "="*60)
print(" VYHODNOCENÍ SIMULACE")
print("="*60)

# ------------------------------------------------------------------------------
# 1. VÝPOČET ENERGETICKÉ BILANCE (High-Precision)
# ------------------------------------------------------------------------------

# A. Vstupní energie (Integrál výkonu)
try:
    E_input_J = np.trapezoid(power_log, cas_plot)
except AttributeError:
    E_input_J = np.trapz(power_log, cas_plot)

E_target_J = E_input_J * beta

# B. Energie v oceli (Integrál přes uzly a segmenty)
E_stored_global_steel_J = 0.0

for i in range(n_segments):
    seg = segments[i]
    energ_segmentu = 0.0
    for j in range(N_nodes):
        T_node_end = T_matrix[i, j]
        # Průměrné cp pro interval ohřevu
        T_avg = (T_node_end + T_olej_inlet) / 2
        _, cp_avg, _ = get_steel_props(T_avg)
        
        vol_node = seg['S_seg'] * dx 
        m_node = vol_node * rho_s_ref
        
        energ_segmentu += m_node * cp_avg * (T_node_end - T_olej_inlet)
    E_stored_global_steel_J += energ_segmentu

E_stored_global_steel_J *= n_pairs

# C. Energie v oleji
E_cooled_J = E_oil_removed_cum_J

# D. Chyba
E_accounted_J = E_stored_global_steel_J + E_cooled_J
diff = E_target_J - E_accounted_J
err_percent = 0.0
if E_target_J > 0:
    err_percent = abs(diff / E_target_J) * 100

validation_status = "PASS" if err_percent < 5.0 else "WARNING"

# ------------------------------------------------------------------------------
# 2. VÝPIS DO KONZOLE
# ------------------------------------------------------------------------------

print(f"1. VSTUP: Energie od motoru (upravená o Betu):  {E_target_J/1000:.1f} kJ")
print("-" * 40)
print(f"2. NALEZENO: Energie uložená v oceli:           {E_stored_global_steel_J/1000:.1f} kJ")
print(f"3. ODVEDENO: Energie odnesená olejem:           {E_cooled_J/1000:.1f} kJ")
print(f"   SOUČET (Nalezeno + Odvedeno):                {E_accounted_J/1000:.1f} kJ")
print("-" * 40)
print(f"ROZDÍL (Chyba modelu):                          {diff/1000:.1f} kJ")
print(f"PROCENTUÁLNÍ CHYBA:                             {err_percent:.2f} %")
print(f"-> VERDIKT: {validation_status}")

print("\n" + "="*60)
print(" SOUHRNNÉ VÝSLEDKY")
print("="*60)

max_oil = max(oil_temp_out_log) if len(oil_temp_out_log) > 0 else T_olej_inlet

print(f"Maximální teplota (Vnější R):  {max(T_surf_R_out_log):.1f} °C")
print(f"Teplota oleje na výstupu (Max):{max_oil:.1f} °C")
print(f"Špičkový tepelný tok (q):      {max_q_net_rec / 1e6:.2f} MW/m²")

if CHLAZENI_TYP == "ANALYTIC_RPM":
    print(f"Režim: SAMONASÁVÁNÍ (Průtok dle RPM)")
elif CHLAZENI_TYP == "ANALYTIC_FIXED":
    print(f"Režim: FIXNÍ PRŮTOK ({q_total_lmin_fixed} L/min)")
elif CHLAZENI_TYP == "ANALYTIC_FIX_TEMPERATURE":
    print(f"Režim: KONSTANTNÍ TEPLOTA OLEJE (Validace 1D)")

# ------------------------------------------------------------------------------
# 3. VYKRESLENÍ GRAFŮ
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

# 1. Teplotní gradient (střed)
axs[0].plot(cas_plot, T_surf_mid_log, 'r-', label='Povrch (Střední R)')
axs[0].plot(cas_plot, T_core_mid_log, 'b--', label='Jádro (Střední R)')
axs[0].set_ylabel('Teplota [°C]', fontweight='bold')
axs[0].set_title(f'1. Teplotní gradient (na středním poloměru)', fontsize=12)
axs[0].grid(True, alpha=0.5); axs[0].legend()

# 2. Zátěž
axs[1].set_ylabel('Moment [Nm]', color='green', fontweight='bold')
axs[1].plot(cas_plot, torque_log, 'g-', label='Moment')
axs[1].tick_params(axis='y', labelcolor='green')
axs[1].grid(True, alpha=0.5)
ax2 = axs[1].twinx()
ax2.set_ylabel('Výkon [kW]', color='orange', fontweight='bold')
ax2.plot(cas_plot, [p/1000 for p in power_log], 'orange', linestyle='--', label='Výkon')
ax2.tick_params(axis='y', labelcolor='orange')
axs[1].set_title('2. Zatížení spojky', fontsize=12)

# 3. Chlazení h
axs[2].plot(cas_plot, h_inner_log, 'b:', label='h - Vnitřní R')
axs[2].plot(cas_plot, h_mid_log, 'b-', label='h - Střední R')
axs[2].plot(cas_plot, h_outer_log, 'b--', label='h - Vnější R')
axs[2].set_ylabel('h [W/m2K]', color='blue', fontweight='bold')
axs[2].set_title(f'3. Součinitel h (Režim: {CHLAZENI_TYP})', fontsize=12)
axs[2].grid(True, alpha=0.5); axs[2].legend()

# 4. Rozložení teploty po poloměru
axs[3].plot(cas_plot, T_surf_R_in_log, 'g:', label='Vnitřní R')
axs[3].plot(cas_plot, T_surf_R_mid_log, 'r-', label='Střední R')
axs[3].plot(cas_plot, T_surf_R_out_log, 'purple', linestyle='--', label='Vnější R')
axs[3].set_ylabel('Teplota [°C]', fontweight='bold')
axs[3].set_xlabel('Čas [s]', fontweight='bold')
axs[3].set_title('4. Rozložení teploty po poloměru', fontsize=12)
axs[3].grid(True, alpha=0.5); axs[3].legend()

# Info box
valid_color = 'green' if validation_status == "PASS" else 'red'
info_text = (f"BILANCE: {validation_status}\nChyba: {err_percent:.2f}%")
axs[3].text(0.02, 0.95, info_text, transform=axs[3].transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=valid_color))

plt.tight_layout()
plt.show()
