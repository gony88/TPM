#podobné s 2D_ochlazevona.py (prace)
# ==============================================================================
# ČÁST 1: IMPORTY A UŽIVATELSKÉ NASTAVENÍ
# ==============================================================================

import numpy as np                  # Knihovna pro matice a matematické výpočty (jádro výpočtů)
import matplotlib.pyplot as plt     # Knihovna pro kreslení grafů
import pandas as pd                 # Knihovna pro práci s Excelem (načtení mapy motoru)
from scipy.interpolate import interp1d # Funkce pro interpolaci (aby mapa motoru byla spojitá)

# ==============================================================================
#                       1. UŽIVATELSKÉ NASTAVENÍ (KONFIGURACE)
# ==============================================================================
# Zde nastavujete vstupy. Co zde změníte, to ovlivní celý fyzikální model.

# --- A. NASTAVENÍ CHLAZENÍ ---
# "ANALYTIC" = Model počítá h dynamicky (v závislosti na otáčkách a teplotě oleje).
# "NO_COOLING" = Vypne chlazení (pro srovnání "worst case").
CHLAZENI_TYP = "NO_COOLING"   # <--- ZDE VOLÍTE REŽIM

# Započítat zmenšení plochy o drážky? (True = menší plocha = větší teplo)
INCLUDE_AREA_REDUCTION = False

# --- B. ODBĚR VÝKONU (PT0 / Hydraulika) ---
# Kolik kW sebere čerpadlo nástavby, než se výkon dostane ke spojce.
P_auxiliary_load_kW = 0.0

# --- C. TEPLOTA OLEJE ---
# Teplota oleje, který přitéká do hřídele (vstup do prvního segmentu).
T_olej_inlet = 70.0  # [°C]

# --- D. ROZJEZD VE SVAHU ---
# Pokud True, simuluje se, že auto chvíli stojí na brzdě/spojce (Hill Hold).
ENABLE_HILL_START = False 
t_hold       = 1.0     # [s] Jak dlouho se drží v kopci
n_motor_hold = 1800.0  # [rpm] Otáčky motoru při držení

# --- E. PRŮBĚH OTÁČEK (Simulace řidiče) ---
# Tvarování křivky otáček (1.0 = přímka, >1.0 = prohnutá křivka).
RPM_SHAPE_FACTOR = 1.0 

n_motor_start = 1200.0 # [rpm] Otáčky na začátku rozjezdu
n_motor_end   = 1200.0 # [rpm] Otáčky na konci prokluzu
n_motor_idle  = 1200.0 # [rpm] Volnoběh po sepnutí

n_slip_start  = 1200.0 # [rpm] Rozdíl otáček na začátku (Motor - Kola)
n_slip_end    = 0.0    # [rpm] Konec prokluzu (spojka sepnuta)

# --- F. ČASOVÁNÍ CYKLU ---
n_cyklu = 1          # Kolikrát se má rozjezd opakovat
t_zab   = 1.5      # [s] Doba trvání prokluzu (jak dlouho řidič pouští spojku)
t_pauza = 1.0        # [s] Doba chlazení po sepnutí

# --- G. GEOMETRIE A SEGMENTACE (Srdce 2D modelu) ---
# Rozdělení lamely na mezikruží (prstence), abychom viděli rozdíl Vnitřek vs. Vnějšek.
n_segments = 10         # Počet segmentů (10 je optimum)
r_out = 0.124           # [m] Vnější poloměr obložení
r_in = 0.0875           # [m] Vnitřní poloměr obložení
tloustka_oceli = 0.006  # [m] Tloušťka ocelového jádra lamely

# --- H. HYDRAULIKA ---
# Kolik oleje teče skrz lamelu (chladicí průtok).
q_total_lmin = 5.0      # [L/min]

# Logická pojistka: Pokud nehceme Hill Start, nastavíme čas držení na 0.
if not ENABLE_HILL_START: 
    t_hold = 0.0
# ==============================================================================
# ČÁST 2: DEFINICE FYZIKÁLNÍCH FUNKCÍ A PŘÍPRAVA GEOMETRIE
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. VLASTNOSTI MATERIÁLŮ
# ------------------------------------------------------------------------------

def get_steel_props(T_celsius):
    """
    Vrátí vlastnosti oceli pro danou teplotu.
    Ocel není konstantní - s teplotou mění své chování.
    """
    # Oříznutí teploty (clip), aby nám rovnice "neulétly" mimo reálné hodnoty (20-1000°C)
    T = np.clip(T_celsius, 20.0, 1000.0)
    
    # Tepelná vodivost k [W/m.K]:
    # Schopnost vést teplo. U oceli klesá s rostoucí teplotou (hůře vede).
    k = 54.0 - 0.028 * T
    
    # Měrná tepelná kapacita cp [J/kg.K]:
    # Schopnost akumulovat teplo. U oceli roste s teplotou (pojme více energie).
    c_p = 450.0 + 0.28 * T
    
    # Hustota rho [kg/m3]: Považujeme za konstantu.
    rho = 7850.0
    
    return k, c_p, rho

def get_oil_viscosity(T_oil):
    """ 
    Vrátí kinematickou viskozitu oleje [m2/s] podle jeho teploty.
    Tohle je klíčové pro 2D model:
    - Studený olej je hustý (velká viskozita) -> teče laminárně -> špatně chladí.
    - Horký olej je řídký -> teče turbulentně -> lépe chladí.
    """
    # Interpolujeme mezi dvěma známými body (typ ATF olej):
    # 40°C -> 30e-6 m2/s (30 cSt)
    # 100°C -> 7e-6 m2/s (7 cSt)
    return np.interp(T_oil, [40, 100], [30e-6, 7e-6])

# ------------------------------------------------------------------------------
# 2. VÝPOČET CHLAZENÍ (Hydrodynamika)
# ------------------------------------------------------------------------------

def get_cooling_analytical_local(rpm, T_oil_local, r_segment_mid, Dh):
    """
    Spočítá součinitel přestupu tepla 'h' [W/m2K] pro konkrétní místo na lamele.
    
    Vstupy:
      rpm:           Otáčky motoru (zdroj odstředivé síly, která žene olej).
      T_oil_local:   Teplota oleje v daném místě (už ohřátá od předchozích segmentů).
      r_segment_mid: Poloměr, kde zrovna počítáme (čím větší R, tím rychlejší tok).
      Dh:            Hydraulický průměr drážky (velikost kanálku).
    """
    # Pokud se motor netočí, je tam jen minimální přirozená konvekce.
    if rpm < 10: return 50.0 
    
    # Konstanty oleje (hustota, vodivost, kapacita)
    rho = 850.0; lam_oil = 0.14; c_oil = 2000.0
    
    # A. Získáme viskozitu pro aktuální teplotu oleje
    nu = get_oil_viscosity(T_oil_local)
    
    # B. Rychlost toku oleje v drážce
    # Předpoklad: Olej je hnán odstředivou silou. Rychlost v = omega * r.
    # To znamená, že na vnějším okraji teče olej rychleji než uvnitř!
    omega = rpm * (2 * np.pi / 60)
    v_oil = omega * r_segment_mid
    
    # C. Reynoldsovo číslo (Re) - Určuje, zda je tok klidný nebo divoký (turbulentní)
    Re = (v_oil * Dh) / nu
    
    # D. Prandtlovo číslo (Pr) - Vlastnost tekutiny
    Pr = (rho * c_oil * nu) / lam_oil
    
    # E. Nusseltovo číslo (Nu) - Bezrozměrné číslo přestupu tepla
    # Používáme korelaci Dittus-Boelter pro turbulentní tok v trubce.
    if Re < 100: Re = 100  # Ošetření proti dělení nulou nebo nesmyslům
    Nu = 0.023 * (Re**0.8) * (Pr**0.3)
    
    # F. Výsledné h (převedení Nu zpět na Watty)
    h_pipe = (Nu / Dh) * lam_oil
    
    # G. Enhancement Factor (Zvýšení účinnosti)
    # Drážky na lamele nejsou hladké trubky. Jsou drsné, krátké a olej se tam míchá.
    # Proto tabulkovou hodnotu zvyšujeme 2x.
    enhancement_factor = 2.0 
    return h_pipe * enhancement_factor

# ------------------------------------------------------------------------------
# 3. NAČTENÍ DAT MOTORU
# ------------------------------------------------------------------------------

def load_engine_map(filename='motor_data.xlsx'):
    """ Načte Excel s charakteristikou motoru (RPM vs Moment). """
    try:
        df = pd.read_excel(filename)
        rpm_data = df['RPM'].values; torque_data = df['Torque'].values
    except FileNotFoundError:
        print(f"INFO: Soubor nenalezen, používám demo data motoru.")
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
    
    # Vytvoří funkci, která umí dopočítat moment pro jakékoliv otáčky
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data

# ==============================================================================
#                       3. INICIALIZACE A KONSTANTY
# ==============================================================================

# Načtení mapy
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()

# Získání vlastností oceli pro referenční teplotu (70°C)
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)

# Vlastnosti třecího materiálu (papír/karbon)
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2

# Výpočet koeficientu BETA (Rozdělení tepla)
# Beta určuje, kolik % tepla vsákne ocel a kolik obložení.
# Počítá se z tepelných jímavostí (sqrt(k*rho*c)).
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

# Konstanty oleje (statické, pro bilance)
c_oil = 2000.0; lambda_oil = 0.14; rho_oil = 850.0

# --- GEOMETRIE DRÁŽEK (WAFFLE) ---
# Potřebujeme spočítat "Hydraulický průměr" (Dh), což je efektivní průměr kanálku.
sirka_drazky = 0.0015    # [m]
hloubka_drazky = 0.0002  # [m]
S_tok = sirka_drazky * hloubka_drazky          # Průřez kanálku
O_tok = 2 * (sirka_drazky + hloubka_drazky)    # Obvod kanálku
Dh = 4 * S_tok / O_tok   # Klíčový parametr pro výpočet Reynoldsova čísla

ratio_groove = 0.06      # 6% plochy jsou drážky
n_pairs = 14             # Počet třecích dvojic (lamel)

# ==============================================================================
# ČÁST 3: PŘÍPRAVA 2D SÍTĚ A ROZDĚLENÍ ZÁTĚŽE
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. GENERACE SEGMENTŮ (Soustředná mezikruží)
# ------------------------------------------------------------------------------

# Vytvoříme hranice poloměrů od r_in do r_out.
# Pokud máme 10 segmentů, potřebujeme 11 hranic.
radii_boundaries = np.linspace(r_in, r_out, n_segments + 1)
segments = [] # Seznam, kam si uložíme data o každém prstýnku

for i in range(n_segments):
    # Určení geometrie pro i-tý prstenec
    r1 = radii_boundaries[i]     # Vnitřní poloměr tohoto kousku
    r2 = radii_boundaries[i+1]   # Vnější poloměr tohoto kousku
    r_mid = (r1 + r2) / 2        # Střední poloměr (použijeme pro výpočet rychlosti oleje)
    
    # Celková plocha mezikruží tohoto segmentu: S = pi * (R^2 - r^2)
    S_seg = np.pi * (r2**2 - r1**2)
    
    # Rozdělení plochy na "Třecí" a "Chladicí" (podle drážek)
    if INCLUDE_AREA_REDUCTION:
        # Pokud drážky ubírají plochu, teplo jde do menšího kusu oceli (bude teplejší)
        S_heat = S_seg * (1 - ratio_groove) 
    else:
        # Standard: Teplo se rozpočítá na celou plochu (konzervativní pro materiál)
        S_heat = S_seg
        
    S_cool = S_seg * ratio_groove # Olej teče jen plochou drážek
    
    # Uložení do seznamu (slovník vlastností)
    segments.append({
        'id': i,
        'r_mid': r_mid,
        'S_seg': S_seg,
        'S_heat': S_heat,
        'S_cool': S_cool,
        'r1': r1,
        'r2': r2
    })

# ------------------------------------------------------------------------------
# 2. ROZDĚLENÍ MOMENTU (Uniform Pressure - Nová spojka)
# ------------------------------------------------------------------------------
# Tady definujeme fyziku rozložení tlaku.
# Pro novou spojku (Uniform Pressure) roste moment s TŘETÍ MOCNINOU poloměru.
# (Protože roste plocha * roste rameno síly * roste rychlost).

denom_torque = (r_out**2 - r_in**2) # Jmenovatel pro normování

for seg in segments:
    # Čitatel pro konkrétní segment
    numerator = (seg['r2']**2 - seg['r1']**2)
    
    # Torque Factor říká: "Kolik procent celkového momentu nese tento prstenec?"
    # Vnější prstence budou mít tento faktor vyšší -> větší q_gen -> vyšší teplota.
    seg['torque_factor'] = numerator / denom_torque

# ------------------------------------------------------------------------------
# 3. PŘÍPRAVA PRŮTOKU OLEJE
# ------------------------------------------------------------------------------

# Přepočet L/min na kg/s (SI jednotky pro fyzikální rovnice)
# (6 l/min) / 60 = 0.1 l/s -> / 1000 = 0.0001 m3/s -> * 850 kg/m3 = kg/s
mdot_total_kg_s = (q_total_lmin / 60 / 1000) * rho_oil

# Klíčové: Olej teče "sériově" skrz celou spojku, ale dělí se mezi jednotlivé třecí plochy.
# Máme n_pairs lamel. Každou mezerou tedy teče jen 1/14 celkového oleje.
mdot_per_surface = mdot_total_kg_s / n_pairs

# ------------------------------------------------------------------------------
# 4. PŘÍPRAVA 1D SOLVERŮ (Matice teplot)
# ------------------------------------------------------------------------------

# Nastavení sítě v tloušťce materiálu (hloubka)
L = tloustka_oceli / 2   # Počítáme jen polovinu tloušťky (symetrie od středu)
N_nodes = 50             # Počet uzlů v tloušťce (čím víc, tím přesnější, ale pomalejší)
dx = L / (N_nodes - 1)   # Vzdálenost mezi uzly [m]

# HLAVNÍ MATICE TEPLOT [Segmenty x Hloubka]
# Toto je paměť simulace.
# Řádek 0 = Teploty vnitřního prstence (od povrchu ke středu)
# Řádek 9 = Teploty vnějšího prstence
# Na začátku mají všechny body teplotu oleje (70°C)
T_matrix = np.full((n_segments, N_nodes), T_olej_inlet)

# Výpis do konzole pro kontrolu nastavení
print("-" * 60)
print(f"INFO: SPUŠTĚNÍ 2D SIMULACE (Multi-Segment 1D + Validace)")
print(f"  * Počet segmentů (r):       {n_segments}")
print(f"  * Režim chlazení:           {CHLAZENI_TYP}")
print(f"  * Model zatížení:           UNIFORM PRESSURE (r^3 -> Vnějšek topí víc)")
print(f"  * Průtok oleje:             {q_total_lmin} L/min")
print("-" * 60)

# ==============================================================================
# ČÁST 4: HLAVNÍ SIMULAČNÍ SMYČKA (KINEMATIKA A VÝKON)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. NASTAVENÍ ČASOVÉHO KROKU (dt)
# ------------------------------------------------------------------------------
# Aby numerický výpočet (FDM) nevybuchl (nezačal oscilovat do nekonečna),
# musí být časový krok 'dt' dostatečně malý.
# Používáme Courantovo kritérium stability (CFL podmínka).

t_cyklus = t_hold + t_zab + t_pauza  # Délka jednoho cyklu rozjezdu
t_total = n_cyklu * t_cyklus         # Celkový čas simulace

# ------------------------------------------------------------------------------
# AUTOMATICKÝ VÝPOČET STABILITY (SAFE MODE)
# ------------------------------------------------------------------------------
# 1. Zjistíme "nejrychlejší" možnou difuzi (studená ocel 20°C vede nejlépe)
k_fast, c_fast, rho_fast = get_steel_props(20.0) 
alpha_max = k_fast / (rho_fast * c_fast) # Nejvyšší tepelná difuzivita

# 2. Vypočítáme kritický časový krok pro zvolené N
# Podmínka stability: dt <= 0.5 * dx^2 / alpha
dt_critical = 0.5 * (dx**2) / alpha_max

# 3. Nastavíme bezpečný krok (90 % kritického)
dt = 0.9 * dt_critical 

print(f"INFO: Automaticky vypočten časový krok dt = {dt:.6f} s")

# ------------------------------------------------------------------------------
# 2. PŘÍPRAVA PROMĚNNÝCH PRO UKLÁDÁNÍ DAT (LOGOVÁNÍ)
# ------------------------------------------------------------------------------
# Do těchto seznamů budeme v průběhu času ukládat výsledky, abychom z nich
# na konci vykreslili grafy.

cas_plot = []         # Časová osa

# Graf 1: Teplota uprostřed lamely (jako reprezentativní vzorek)
T_surf_mid_log = []   # Povrchová teplota (Node 0)
T_core_mid_log = []   # Teplota jádra (Node -1)

# Graf 2: Výkon motoru a Moment
torque_log = []       # Moment [Nm]
power_log = []        # Výkon [W]
rpm_log = []          # Otáčky motoru [rpm]

# Graf 3: Součinitel přestupu tepla 'h' na různých místech
h_inner_log = []      # h na vnitřním poloměru (pomalý tok)
h_mid_log = []        # h uprostřed
h_outer_log = []      # h na vnějším poloměru (rychlý tok)

# Graf 4: Rozložení teploty po poloměru (To, co vás nejvíc zajímá)
T_surf_R_in_log = []  # Teplota vnitřního kroužku
T_surf_R_mid_log = [] # Teplota středního kroužku
T_surf_R_out_log = [] # Teplota vnějšího kroužku

# Pomocné indexy, abychom věděli, který segment je který
idx_mid = n_segments // 2     # Index prostředního segmentu (např. 5)
idx_out = n_segments - 1      # Index posledního segmentu (vnější okraj)

# Proměnné pro sledování maxim (pro závěrečnou statistiku)
max_torque_rec = 0.0
max_power_net_rec = 0.0
max_q_net_rec = 0.0

# ### NOVÉ PRO BILANCI ###
# Sem budeme sčítat veškerou energii, kterou olej odnesl pryč.
E_oil_removed_cum_J = 0.0 

# Inicializace proměnných smyčky
t = 0.0
step = 0

# Uložení počátečního stavu (t=0) do grafů
cas_plot.append(0.0)
T_surf_mid_log.append(T_olej_inlet); T_core_mid_log.append(T_olej_inlet)
torque_log.append(0.0); power_log.append(0.0); rpm_log.append(0.0)
h_inner_log.append(0.0); h_mid_log.append(0.0); h_outer_log.append(0.0)
T_surf_R_in_log.append(T_olej_inlet); T_surf_R_mid_log.append(T_olej_inlet); T_surf_R_out_log.append(T_olej_inlet)

print("... Simulace běží ...")

# ------------------------------------------------------------------------------
# 3. START HLAVNÍ SMYČKY
# ------------------------------------------------------------------------------
while t < t_total:
    
    # Kde se nacházíme v rámci jednoho cyklu? (0 až t_cyklus)
    t_local = t % t_cyklus
    
    # -----------------------------------------------------------
    # A. KINEMATIKA (Otáčky a Moment)
    # -----------------------------------------------------------
    # Zde definujeme chování řidiče/vozidla.
    
    if t_local < t_hold:
        # FÁZE 1: HILL HOLD (Předzáběr v kopci)
        # Motor má konstantní otáčky, auto stojí, spojka prokluzuje.
        rpm_engine = n_motor_hold
        rpm_slip = n_motor_hold   # Slip = Engine (protože výstup stojí)
        torque_total = get_torque_from_rpm(rpm_engine) # Moment z mapy motoru
        
    elif t_local < (t_hold + t_zab):
        # FÁZE 2: LAUNCH (Samotný rozjezd)
        # Otáčky se mění z n_start na n_end.
        t_in = t_local - t_hold       # Čas od začátku rozjezdu
        ratio_lin = t_in / t_zab      # Lineární poměr (0 až 1)
        
        # Aplikace tvarového faktoru (zakřivení průběhu otáček)
        ratio = ratio_lin ** RPM_SHAPE_FACTOR
        
        # Interpolace otáček motoru
        rpm_engine = n_motor_start + (n_motor_end - n_motor_start) * ratio
        # Interpolace prokluzu (klesá k nule)
        rpm_slip = n_slip_start * (1 - ratio)
        
        # Moment z mapy motoru pro aktuální otáčky
        torque_total = get_torque_from_rpm(rpm_engine)
        
    else:
        # FÁZE 3: PAUZA / JÍZDA (Spojka sepnuta nebo rozepnuta)
        rpm_engine = n_motor_idle
        rpm_slip = 0.0            # Žádný prokluz = Žádné teplo
        torque_total = 0.0        # Žádný přenášený moment (nebo sepnuto beze ztrát)

    # -----------------------------------------------------------
    # B. VÝPOČET CELKOVÉHO VSTUPNÍHO VÝKONU
    # -----------------------------------------------------------
    
    # Úhlová rychlost prokluzu [rad/s]
    omega_slip = rpm_slip * 2 * np.pi / 60
    
    # Celkový hrubý výkon [W] = Moment * Omega
    P_gross = torque_total * omega_slip
    
    # Čistý výkon do spojky [W] (ponížený o hydrauliku/nástavbu)
    # Funkce max(0, ...) zajistí, že výkon nebude záporný.
    P_net_total = max(0.0, P_gross - (P_auxiliary_load_kW * 1000))

    # -----------------------------------------------------------
    # C. SMYČKA PŘES SEGMENTY (Hydraulika + Teplo)
    # -----------------------------------------------------------
    # Toto je jádro 2D modelu. Olej teče sériově:
    # Vstup -> Seg 0 -> Seg 1 -> ... -> Seg 9 -> Výstup
    
    # Reset teploty oleje na vstupu (vždy přitéká čerstvý ze skříně)
    T_oil_current = T_olej_inlet
    
    # Proměnné pro logování hodnot v tomto časovém kroku
    h_curr_in = 0; h_curr_mid = 0; h_curr_out = 0
    max_q_step = 0.0
    
    # Akumulátor pro validaci: Kolik tepla olej odnesl v tomto kroku?
    P_removed_oil_step_W = 0.0
    
    for i in range(n_segments):
        seg = segments[i] # Načteme parametry aktuálního prstence
        
        # --- C1. GENERACE TEPLA (q_gen) ---
        # Kolik z celkového výkonu připadne na tento prstenec?
        # (Vnější prstence berou víc díky 'torque_factor')
        P_seg = P_net_total * seg['torque_factor']
        
        # Přepočet na tepelný tok [W/m2]
        # Dělíme plochou páru (S_heat) a počtem párů (n_pairs).
        # Násobíme Betou (část tepla jde do papíru, část do oceli).
        q_gen = (P_seg / (n_pairs * seg['S_heat'])) * beta
        
        # --- C2. VÝPOČET CHLAZENÍ (h) ---
        # Určíme součinitel přestupu tepla pro TENTO segment.
        # Používáme LOKÁLNÍ teplotu oleje (už se mohla ohřát) a LOKÁLNÍ poloměr.
        
        if CHLAZENI_TYP == "ANALYTIC":
            h_seg = get_cooling_analytical_local(rpm_engine, T_oil_current, seg['r_mid'], Dh)
        elif CHLAZENI_TYP == "FLOW_LIMIT":
            # Zjednodušený model: h je limitováno jen tepelnou kapacitou průtoku
            h_seg = (mdot_per_surface * c_oil) / seg['S_cool']
        elif CHLAZENI_TYP == "STATIC":
            h_seg = 800.0 # Fixní hodnota pro ladění
        else:
            h_seg = 0.0   # Bez chlazení
            
        # Uložení h pro grafy (jen vybrané segmenty)
        if i == 0: h_curr_in = h_seg
        if i == idx_mid: h_curr_mid = h_seg
        if i == idx_out: h_curr_out = h_seg
        
        # --- C3. TEPELNÁ BILANCE POVRCHU ---
        T_surf = T_matrix[i, 0] # Aktuální teplota povrchu v tomto segmentu
        
        # Newtonův ochlazovací zákon: q_cool = h * (T_ocel - T_olej)
        q_cool = h_seg * (T_surf - T_oil_current)
        
        # Čistý tok do materiálu: Co vyrobím mínus co uchladím
        q_net = q_gen - q_cool
        
        # Sledování maxima pro statistiku
        if q_net > max_q_step: max_q_step = q_net
        
        # --- C4. OHŘEV OLEJE (Hydraulická vazba) ---
        # Teplo, které jsme odebrali oceli (q_cool), se musí uložit do oleje.
        # Q_absorbed [W] = tok [W/m2] * plocha [m2]
        Q_absorbed = q_cool * seg['S_heat']
        
        # O kolik se olej ohřeje? dT = Q / (m_dot * c_p)
        dT_oil = Q_absorbed / (mdot_per_surface * c_oil)
        
        # Aktualizace teploty oleje: 
        # Do DALŠÍHO segmentu už poteče teplejší olej!
        T_oil_current += dT_oil
        
        # VALIDACE: Přičteme toto teplo k celkové sumě odvedeného tepla
        P_removed_oil_step_W += Q_absorbed
        
        # --- C5. FDM SOLVER (Vedení tepla dovnitř oceli) ---
        # Spočítáme, jak se teplo šíří z povrchu do středu lamely v tomto segmentu.
        
        # 1. Zjistíme vlastnosti oceli pro aktuální rozložení teplot (nelinearita)
        k_vec, cp_vec, rho_val = get_steel_props(T_matrix[i, :])
        alpha_vec = k_vec / (rho_val * cp_vec) # Teplotní difuzivita
        
        # 2. Explicitní schéma (Finite Difference Method)
        T_old = T_matrix[i, :]     # Teploty v minulém kroku
        T_new_seg = np.copy(T_old) # Pole pro nové teploty
        
        # A) Vnitřní uzly (vedení tepla materiálem)
        # T_new = T_old + alpha * dt/dx^2 * (T_levy - 2*T_stred + T_pravy)
        T_new_seg[1:-1] = T_old[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T_old[2:] - 2*T_old[1:-1] + T_old[:-2])
        
        # B) Povrchový uzel (Okrajová podmínka: Tepelný tok q_net)
        T_new_seg[0] = T_old[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T_old[0] - T_old[1]) / dx)
        
        # C) Středový uzel (Okrajová podmínka: Symetrie/Izolace)
        # Teplo dál neproudí (protože z druhé strany je stejná lamela)
        T_new_seg[-1] = T_old[-1] + alpha_vec[-1] * dt / dx**2 * (T_old[-2] - T_old[-1])
        
        # Uložení vypočtených teplot zpět do hlavní paměti
        T_matrix[i, :] = T_new_seg[:]

    # -----------------------------------------------------------
    # D. AKTUALIZACE GLOBÁLNÍCH DAT A ČASU
    # -----------------------------------------------------------
    
    # VALIDACE: P_removed_oil_step_W je teplo z jedné poloviny lamely. 
    # Celá spojka má n_pairs lamel. Vynásobíme a přičteme k celkové energii (Jouly).
    # Energie [J] = Výkon [W] * čas [s]
    E_oil_removed_cum_J += P_removed_oil_step_W * n_pairs * dt

    # Uložení maxim (pro závěrečný report)
    if torque_total > max_torque_rec: max_torque_rec = torque_total
    if P_net_total > max_power_net_rec: max_power_net_rec = P_net_total
    if max_q_step > max_q_net_rec: max_q_net_rec = max_q_step
    
    # Posun v čase
    t += dt
    step += 1
    
    # Logování dat pro grafy (ukládáme jen každý 200. krok, aby se nezahltila paměť)
    if step % 200 == 0:
        cas_plot.append(t)
        
        # Ukládáme data pro STŘEDNÍ segment (reprezentativní)
        T_surf_mid_log.append(T_matrix[idx_mid, 0])
        T_core_mid_log.append(T_matrix[idx_mid, -1])
        
        # Ukládáme globální veličiny
        torque_log.append(torque_total)
        power_log.append(P_net_total)
        rpm_log.append(rpm_engine)
        
        # Ukládáme 'h' na různých místech
        h_inner_log.append(h_curr_in)
        h_mid_log.append(h_curr_mid)
        h_outer_log.append(h_curr_out)
        
        # Ukládáme Povrchové teploty na různých radiusech (TO DŮLEŽITÉ)
        # T_matrix[0,0] = Vnitřní, T_matrix[-1,0] = Vnější
        T_surf_R_in_log.append(T_matrix[0, 0])
        T_surf_R_mid_log.append(T_matrix[idx_mid, 0])
        T_surf_R_out_log.append(T_matrix[idx_out, 0])

        # ==============================================================================
# ČÁST 6: VYHODNOCENÍ, VALIDACE A VYKRESLENÍ
# ==============================================================================

print("\n" + "="*60)
print(" VYHODNOCENÍ SIMULACE")
print("="*60)

# ------------------------------------------------------------------------------
# 1. VÝPOČET ENERGETICKÉ BILANCE (VALIDACE)
# ------------------------------------------------------------------------------
# Zde ověřujeme, zda model "neztratil" energii vlivem numerických chyb.

# A. Celková energie dodaná motorem (Joule)
# Spočítáme plochu pod křivkou výkonu (integrál P * dt).
try:
    # Pro novější verze NumPy (2.0+)
    E_input_J = np.trapezoid(power_log, cas_plot)
except AttributeError:
    # Pro starší verze NumPy
    E_input_J = np.trapz(power_log, cas_plot)

# B. Cílová energie pro ocel (Joule)
# Z celkové energie motoru jde část do papíru a část do oceli (určeno Betou).
# Náš model počítá jen ocel, takže musíme vstup ponížit o papír.
E_target_J = E_input_J * beta

# C. Energie skutečně nalezená v oceli na konci simulace (Joule)
# Projdeme všechny segmenty a spočítáme: m * c * deltaT
E_stored_global_steel_J = 0.0

for i in range(n_segments):
    seg = segments[i]
    # Průměrná teplota celého segmentu (všech uzlů v tloušťce)
    T_final_avg = np.mean(T_matrix[i, :])
    
    # O kolik se tento segment ohřál oproti startu?
    delta_T_seg = T_final_avg - T_olej_inlet
    
    # Hmotnost jednoho segmentu (polovina tloušťky jedné lamely)
    m_seg = seg['S_seg'] * (tloustka_oceli / 2) * rho_s_ref 
    
    # Energie v jedné lamele
    E_stored_global_steel_J += m_seg * c_s_ref * delta_T_seg 

# Vynásobíme počtem párů (celá spojka)
E_stored_global_steel_J *= n_pairs

# D. Energie odnesená olejem (Joule)
# Tuto hodnotu jsme sčítali v každém kroku smyčky (proměnná E_oil_removed_cum_J).
E_cooled_J = E_oil_removed_cum_J

# E. Finální součet (Bilance)
# Mělo by platit: Co mělo přijít (Target) = Co tam zůstalo (Stored) + Co odteklo (Cooled)
E_accounted_J = E_stored_global_steel_J + E_cooled_J

# F. Výpočet chyby
diff = E_target_J - E_accounted_J
err_percent = 0.0
if E_target_J > 0:
    err_percent = abs(diff / E_target_J) * 100

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

# Určení statusu validace
validation_status = "UNKNOWN"
if err_percent < 5.0:
    validation_status = "PASS"
    print("-> VERDIKT: OK (Bilance sedí, model je fyzikálně správný)")
else:
    validation_status = "WARNING"
    print("-> VERDIKT: POZOR (Odchylka > 5%, zkontroluj cp/betu nebo krok dt)")

print("-" * 40)
print(f"Špičkový točivý moment:      {max_torque_rec:.1f} Nm")
# P_gross vs P_net (zde jsou stejné, protože P_aux = 0)
# Pro výpis použijeme max hodnoty z logu
max_p_log = max(power_log) if len(power_log) > 0 else 0
print(f"1. CELKOVÝ TŘECÍ VÝKON (GROSS):  {max_p_log / 1000:.1f} kW")
print(f"   (Teoretické maximum bez odběru)")
print(f"2. ODBĚR NÁSTAVBOU:              {P_auxiliary_load_kW:.1f} kW")
print(f"3. VÝSLEDNÝ VÝKON DO SPOJKY:     {(max_p_log / 1000):.1f} kW")
print(f"   (Poníženo o nástavbu)")

print("\n" + "="*60)
print(f" SOUHRNNÉ VÝSLEDKY")
print("="*60)
# Vypíšeme teplotu na vnějším okraji (tam kde to nejvíc pálí)
print(f"Maximální teplota (Vnější R):  {max(T_surf_R_out_log):.1f} °C")
print(f"Maximální teplota (Střední R):  {max(T_surf_R_mid_log):.1f} °C")
print(f"Maximální teplota (Vnitřní R):  {max(T_surf_R_in_log):.1f} °C")
print(f"Teplota oleje na výstupu:      {T_oil_current:.1f} °C")
print(f"Špičkový tepelný tok (q):      {max_q_net_rec / 1e6:.2f} MW/m²")



# ------------------------------------------------------------------------------
# 3. VYKRESLENÍ GRAFŮ
# ------------------------------------------------------------------------------

# Vytvoření okna se 4 grafy pod sebou
fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

# --- GRAF 1: Teplotní gradient v tloušťce (Střední poloměr) ---
# Ukazuje, jak rychle teplo proniká z povrchu do středu oceli.
axs[0].plot(cas_plot, T_surf_mid_log, 'r-', label='Povrch (Střední R)', linewidth=1.5)
axs[0].plot(cas_plot, T_core_mid_log, 'b--', label='Jádro (Střední R)', linewidth=1.5)
axs[0].set_ylabel('Teplota [°C]', fontweight='bold')
axs[0].set_title(f'1. Teplotní gradient v tloušťce materiálu (na středním poloměru)', fontsize=12)
axs[0].grid(True, alpha=0.5)
axs[0].legend(loc='upper right')

# --- GRAF 2: Zátěž (Moment a Výkon) ---
# Ukazuje vstupní energii do systému.
axs[1].set_ylabel('Moment [Nm]', color='green', fontweight='bold')
line1 = axs[1].plot(cas_plot, torque_log, 'g-', label='Moment', alpha=0.8)
axs[1].tick_params(axis='y', labelcolor='green')
axs[1].grid(True, alpha=0.5)

# Druhá osa Y pro výkon (v kW)
ax2_twin = axs[1].twinx()
ax2_twin.set_ylabel('Výkon [kW]', color='orange', fontweight='bold')
power_kw = [p / 1000 for p in power_log]
line2 = ax2_twin.plot(cas_plot, power_kw, color='orange', linestyle='--', label='Výkon')
ax2_twin.tick_params(axis='y', labelcolor='orange')

# Společná legenda pro obě osy
lns = line1 + line2
labs = [l.get_label() for l in lns]
axs[1].legend(lns, labs, loc='center right')
axs[1].set_title('2. Zatížení spojky (Celkové)', fontsize=12)

# --- GRAF 3: Chlazení (Součinitel h) ---
# Ukazuje, jak se mění účinnost chlazení s otáčkami a poloměrem.
axs[2].plot(cas_plot, h_inner_log, 'b:', label='h - Vnitřní R (Pomalý tok)', alpha=0.7)
axs[2].plot(cas_plot, h_mid_log, 'b-', label='h - Střední R', linewidth=2)
axs[2].plot(cas_plot, h_outer_log, 'b--', label='h - Vnější R (Rychlý tok)', alpha=0.7)
axs[2].set_ylabel('h [W/m2K]', color='blue', fontweight='bold')
axs[2].set_title('3. Součinitel přestupu tepla h na různých poloměrech', fontsize=12)
axs[2].grid(True, alpha=0.5)
axs[2].legend(loc='upper right')

# --- GRAF 4: Rozložení teploty po poloměru ---
# TOTO JE NEJDŮLEŽITĚJŠÍ GRAF VAŠEHO 2D MODELU.
# Ukazuje, o kolik je vnější okraj teplejší než vnitřní.
axs[3].plot(cas_plot, T_surf_R_in_log, color='green', linestyle=':', label='Povrch - Vnitřní R')
axs[3].plot(cas_plot, T_surf_R_mid_log, color='red', linestyle='-', label='Povrch - Střední R')
axs[3].plot(cas_plot, T_surf_R_out_log, color='purple', linestyle='--', label='Povrch - Vnější R (Hotspot)')
axs[3].set_ylabel('Teplota [°C]', fontweight='bold')
axs[3].set_xlabel('Čas [s]', fontweight='bold')
axs[3].set_title('4. Rozložení teploty po poloměru lamely', fontsize=12)
axs[3].grid(True, alpha=0.5)
axs[3].legend(loc='upper right')

# --- PŘIDÁNÍ VÝSLEDKU VALIDACE PŘÍMO DO GRAFU ---
# Vytvoříme malé informační okno v grafu č. 4
valid_color = 'green' if validation_status == "PASS" else 'red'
info_text = (f"BILANCE ENERGIE:\n"
             f"Vstup: {E_target_J/1000:.1f} kJ\n"
             f"Nalezeno: {E_accounted_J/1000:.1f} kJ\n"
             f"Chyba: {err_percent:.2f} %\n"
             f"Status: {validation_status}")

axs[3].text(0.02, 0.95, info_text, transform=axs[3].transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=valid_color, alpha=0.8))

plt.tight_layout()
plt.show()

# KONEC KÓDU
