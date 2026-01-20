import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# ==============================================================================
# NASTAVENÍ MODELU (USER CONFIG)
# ==============================================================================

# 1. TYP CHLAZENÍ
# "STATIC"     = Fixní hodnota (Laminární v drážce, vypočteno na začátku)
# "ANALYTIC"   = Dynamicky se mění dle otáček motoru (Odstředivé čerpadlo + Waffle)
# "FLOW_LIMIT" = Fixní hodnota omezená tepelnou kapacitou průtoku oleje
# "NO_COOLING" = Žádné chlazení (Simulace poruchy / Adiabatický děj)
CHLAZENI_TYP = "ANALYTIC"   # <--- ZMĚŇTE NA "STATIC", "ANALYTIC", "FLOW_LIMIT" NEBO "NO_COOLING"

# 2. VLIV PLOCHY DRÁŽEK NA ZAHŘÍVÁNÍ
# True  = Výkon motoru jde do menší plochy (odečteny drážky) -> Větší teplo
# False = Výkon motoru se rozpočítá na celou plochu (ignoriuje drážky)
INCLUDE_AREA_REDUCTION = True

# --- NOVÉ: DEFINICE OTÁČEK (ZDE JE TA ZMĚNA) ---
# Rozlišujeme, co dělá motor (chlazení/moment) a co dělá prokluz (teplo).

# A. ABSOLUTNÍ OTÁČKY MOTORU (Pro mapu momentu a chlazení)
# Motor se při řazení/rozjezdu nezastavuje!
n_motor_start = 2000.0  # [rpm] Otáčky motoru na začátku
n_motor_end   = 2000.0  # [rpm] Otáčky motoru na konci (třeba držíme plyn)
n_motor_idle  = 800.0   # [rpm] Otáčky motoru v PAUZE (volnoběh)

# B. OTÁČKY PROKLUZU (Pro výpočet tepla P = M * omega_slip)
# Rozdíl rychlostí mezi motorem a převodovkou. Vždy klesá k nule.
n_slip_start  = 2000.0  # [rpm] Počáteční prokluz (např. stojící auto = otáčky motoru)
n_slip_end    = 0.0     # [rpm] Konec prokluzu (vždy 0 = sepnuto)

# ==============================================================================

# --- 1. FUNKCE PRO VLASTNOSTI A CHLAZENÍ ---

def get_steel_props(T_celsius):
    """
    Vrací vlastnosti uhlíkové oceli v závislosti na teplotě.
    """
    T = np.clip(T_celsius, 20.0, 1000.0)
    # 1. Tepelná vodivost k [W/m.K]
    k = 54.0 - 0.028 * T
    # 2. Měrná tepelná kapacita cp [J/kg.K]
    c_p = 450.0 + 0.28 * T
    # 3. Hustota rho [kg/m3]
    rho = 7850.0
    return k, c_p, rho

def get_cooling_analytical(rpm, T_oil_C, geometry):
    """
    Vypočítá h na základě fyziky rotujícího kanálku (odstředivé čerpadlo).
    Používá ABSOLUTNÍ otáčky koše spojky (motoru).
    """
    if rpm < 10: return 50.0 # Minimum při stání
    
    # A. Vlastnosti oleje (ATF)
    rho = 850.0
    # Viskozita klesá s teplotou (30cSt při 40°C -> 7cSt při 100°C)
    nu = np.interp(T_oil_C, [40, 100], [30e-6, 7e-6]) 
    mu = nu * rho
    lam_oil = 0.14
    c_oil = 2000.0
    Pr = (mu * c_oil) / lam_oil

    # B. Geometrie
    r_in, r_out = geometry['r_in'], geometry['r_out']
    Dh = geometry['Dh']
    L = r_out - r_in
    
    # C. Odstředivý tlak (Driving Force)
    omega = rpm * (2 * np.pi / 60)
    dP = 0.5 * rho * omega**2 * (r_out**2 - r_in**2)
    
    # D. Rychlost proudění
    K_shape = 24.0 # Tvarová konstanta
    v_oil = (dP * Dh**2) / (K_shape * mu * L)
    
    # E. Bezrozměrná čísla
    Re = (v_oil * Dh) / nu
    
    # F. Nusseltovo číslo
    if Re < 2300:
        # Laminární
        Nu = 1.86 * (Re * Pr * (Dh/L))**(1/3)
        Nu = max(Nu, 3.66) 
    else:
        # Turbulentní
        Nu = 0.023 * Re**0.8 * Pr**0.4
        
    # G. Výsledné h
    h = (Nu * lam_oil) / Dh
    h *= 1.5 # Waffle Factor
    
    return h

# --- 2. NAČTENÍ DAT Z EXCELU ---

def load_engine_map(filename='motor_data.xlsx'):
    try:
        df = pd.read_excel(filename)
        rpm_data = df['RPM'].values
        torque_data = df['Torque'].values
        print(f"ÚSPĚCH: Načten soubor '{filename}'.")
    except FileNotFoundError:
        print(f"CHYBA: Soubor '{filename}' nenalezen!")
        print("POUŽÍVÁM NÁHRADNÍ DATA PRO UKÁZKU.")
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
    except KeyError:
        print("CHYBA: Excel musí obsahovat sloupce 'RPM' a 'Torque'.")
        raise
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data

# Inicializace mapy motoru
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()

# --- 3. FYZIKÁLNÍ KONSTANTY A VSTUPY ---

# Materiál: Ocel (Separační lamela)
# Zde bereme referenční hodnoty pro výpočet Bety
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)

# Materiál: Třecí obložení
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2

# Výpočet Beta
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

# Vlastnosti oleje (pro statický výpočet)
lambda_oil = 0.14    
T_olej = 70.0        
c_oil = 2000.0       

# Geometrie
r_out = 0.124        
r_in = 0.0875        
tloustka_oceli = 0.004 
T_skrin = 70.0       

# Definice drážek (WAFFLE)
sirka_drazky = 0.0015  
hloubka_drazky = 0.0002 
roztec_drazek = 0.009 

# Hydraulický průměr
S_tok = sirka_drazky * hloubka_drazky
O_tok = 2 * (sirka_drazky + hloubka_drazky)
Dh = 4 * S_tok / O_tok

# Geometrie pro analytickou funkci
geometry_dict = {'r_in': r_in, 'r_out': r_out, 'Dh': Dh}

# Podíl ploch
S_celkova_mezikruzi = np.pi * (r_out**2 - r_in**2)
ratio_groove = 0.06 # % zastoupení drážek (0 = žádné drážky)
S_cooling = S_celkova_mezikruzi * ratio_groove   # Plocha kanálků       
S_contact = S_celkova_mezikruzi * (1 - ratio_groove) # Čistá plocha oceli

# Rozhodnutí, jakou plochu použít pro dělení výkonu (User Config)
if INCLUDE_AREA_REDUCTION:
    S_calc_power = S_contact # Výkon jde do menší plochy (HORKÉ!)
    area_note = "Redukovaná (Odečteny drážky)"
else:
    S_calc_power = S_celkova_mezikruzi # Výkon se "rozlije" i přes díry
    area_note = "Celková (Ignorovány drážky)"

# Výkonové parametry
n_pairs = 14          

# ==============================================================================
# PŘÍPRAVA HODNOT PRO CHLAZENÍ (INICIALIZACE)
# ==============================================================================

h_used_initial = 0.0

# 1. STATIC (Laminární odhad)
Nu_stat = 6.05  
h_static_val = (Nu_stat * lambda_oil) / Dh

# 2. FLOW LIMIT (Kapacita průtoku)
q_total_lmin = 6.0 # [L/min]
mdot_per_surface = (q_total_lmin / 60 / 1000 * 850) / n_pairs 
h_flow_limit_val = (mdot_per_surface * c_oil) / S_cooling

# Logika výběru pro výpis startovacích podmínek
if CHLAZENI_TYP == "STATIC":
    h_used_initial = h_static_val
    print(f"\n--- REŽIM: STATIC ---")
    print(f"Použité fixní h: {h_used_initial:.1f} W/m2K")

elif CHLAZENI_TYP == "FLOW_LIMIT":
    h_used_initial = h_flow_limit_val
    print(f"\n--- REŽIM: FLOW_LIMIT ---")
    print(f"Použité limitní h: {h_used_initial:.1f} W/m2K (Omezeno průtokem {q_total_lmin} L/min)")

elif CHLAZENI_TYP == "ANALYTIC":
    print(f"\n--- REŽIM: ANALYTIC ---")
    print(f"Hodnota h se bude měnit dynamicky dle otáček motoru ({n_motor_start} rpm).")
    # Pro kontrolu vypočteme startovací hodnotu pro MOTOR START
    h_check = get_cooling_analytical(n_motor_start, T_olej, geometry_dict)
    h_used_initial = h_check
    print(f"Startovací odhad h (při {n_motor_start} rpm): {h_check:.1f} W/m2K")

elif CHLAZENI_TYP == "NO_COOLING":
    h_used_initial = 0.0
    print(f"\n--- REŽIM: NO_COOLING ---")
    print(f"Vypnuto chlazení (Adiabatický děj). h = 0 W/m2K")


# --- NOVÉ: VÝPIS VŠECH MOŽNOSTÍ (PRO POROVNÁNÍ) ---
print("-" * 50)
print("PŘEHLED MOŽNÝCH HODNOT CHLAZENÍ (PRO INFO):")
# Pro Analytic musíme spočítat demo hodnotu
h_analytic_demo_val = get_cooling_analytical(n_motor_start, T_olej, geometry_dict)

print(f"  1. STATIC:      {h_static_val:.1f} W/m2K")
print(f"  2. FLOW LIMIT:  {h_flow_limit_val:.1f} W/m2K (při {q_total_lmin} L/min)")
print(f"  3. ANALYTIC:    {h_analytic_demo_val:.1f} W/m2K (při {n_motor_start} rpm)")
print(f"  4. NO COOLING:  0.0 W/m2K")
print("-" * 50)


# --- 4. SIMULACE (FDM 1D) ---

n_cyklu = 1        
t_zab = 1.74            
t_pauza = 5.0    
t_cyklus = t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Diskretizace
L = tloustka_oceli / 2 
N = 50                 
dx = L / (N - 1)       

# Stabilita (podle studené oceli)
k_c, c_c, rho_c = get_steel_props(20.0)
dt = 0.9 * (0.5 * dx**2 / (k_c / (rho_c * c_c)))

# Inicializace
T = np.full(N, T_skrin)
T_new = np.copy(T)

# Pro grafy
cas_plot = []
T_surf_plot = []
T_core_plot = []
h_log = [] # Logujeme i vývoj h

# NOVÉ: Logování momentu a výkonu pro graf
torque_log = []
power_log = []

# Proměnné pro maxima (výpis do terminalu)
max_torque_rec = 0.0
max_power_rec = 0.0
max_q_net_rec = 0.0

t = 0.0
step = 0

print("Startuji simulaci...")

while t < t_total:
    t_local = t % t_cyklus

    # --- KROK 1: URČENÍ OTÁČEK V DANÉM ČASE ---
    if t_local <= t_zab:
        ratio = t_local / t_zab
        
        # A) ABSOLUTNÍ OTÁČKY MOTORU (Pro Moment a Chlazení)
        # Motor se točí pořád!
        rpm_engine_abs = n_motor_start + (n_motor_end - n_motor_start) * ratio
        
        # B) OTÁČKY PROKLUZU (Pro Tepelný Výkon)
        # Rozdíl rychlostí jde vždy k nule
        rpm_slip = n_slip_start * (1 - ratio)
        
    else:
        # Pauza (Volnoběh)
        rpm_engine_abs = n_motor_idle
        rpm_slip = 0.0


    # --- KROK 2: URČENÍ AKTUÁLNÍHO 'h' ---
    if CHLAZENI_TYP == "STATIC":
        h_current = h_static_val
    
    elif CHLAZENI_TYP == "FLOW_LIMIT":
        h_current = h_flow_limit_val
        
    elif CHLAZENI_TYP == "NO_COOLING":
        h_current = 0.0
        
    else: # ANALYTIC
        # Používáme ABSOLUTNÍ otáčky (čerpadlo je hnané motorem)
        T_ref_oil = (T[0] + T_olej) / 2 
        h_current = get_cooling_analytical(rpm_engine_abs, T_ref_oil, geometry_dict)


    # --- KROK 3: VÝPOČET TEPELNÉHO ZDROJE ---
    if t_local <= t_zab:
        # 1. Získání momentu (Z MAPY DLE MOTORU!)
        real_torque = get_torque_from_rpm(rpm_engine_abs)
        if real_torque < 0: real_torque = 0
        
        # 2. Výkon = Moment * SKLUZ
        omega_slip = rpm_slip * 2 * np.pi / 60
        power = real_torque * omega_slip
        
        # 3. Tepelný tok
        q_gen = (power / n_pairs / S_calc_power) * beta
        
        # 4. Chlazení (vždy počítáme jen na ploše drážek -> ratio_groove)
        q_cool = h_current * (T[0] - T_olej) * ratio_groove
        
        q_net = q_gen - q_cool
        
        # Log maxim
        if real_torque > max_torque_rec: max_torque_rec = real_torque
        if power > max_power_rec: max_power_rec = power
        if q_net > max_q_net_rec: max_q_net_rec = q_net

        # Pro graf
        current_torque_plot = real_torque
        current_power_plot = power

    else:
        # Fáze PAUZY
        q_gen = 0.0
        # Stále chladíme (pokud není NO_COOLING, h_current bude > 0 díky volnoběhu)
        q_cool = h_current * (T[0] - T_olej) * ratio_groove
        q_net = -q_cool

        current_torque_plot = 0.0
        current_power_plot = 0.0

    # --- KROK 4: SOLVER ---
    # Aktualizace materiálu
    k_vec, cp_vec, rho_val = get_steel_props(T)
    alpha_vec = k_vec / (rho_val * cp_vec)

    # Vnitřní uzly
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
 
    # Povrch
    T_new[0] = T[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
   
    # Střed
    T_new[-1] = T[-1] + alpha_vec[-1] * dt / dx**2 * (T[-2] - T[-1])

    T[:] = T_new[:]
    t += dt
    step += 1

    if step % 200 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])
        h_log.append(h_current)
        torque_log.append(current_torque_plot)
        power_log.append(current_power_plot)

# --- VÝPIS DO TERMINÁLU ---
print("\n" + "="*60)
print(f" VÝSLEDKY SIMULACE")
print("="*60)
print(f"Koeficient Beta:             {beta:.3f} (-)")
print(f"Plocha lamely (S_calc):      {S_calc_power * 10000:.2f} cm² ({area_note})")
print(f"Hydraulický průměr Dh:       {Dh*1000:.2f} mm")
print("-" * 60)
print(f"Špičkový točivý moment:      {max_torque_rec:.1f} Nm")
print(f"Špičkový výkon (celkový):    {max_power_rec / 1000:.1f} kW")
print(f"Špičkový tepelný tok (q):    {max_q_net_rec / 1e6:.2f} MW/m²")
print(f"MAXIMÁLNÍ TEPLOTA POVRCHU:   {max(T_surf_plot):.1f} °C")
print("="*60)

# --- 5. VYKRESLENÍ (DVA GRAFY) ---

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# GRAF 1: TEPLOTA
ax1.plot(cas_plot, T_surf_plot, 'r-', label='Povrch (x=0)', linewidth=1)
ax1.plot(cas_plot, T_core_plot, 'b-', label='Střed (x=L)', linewidth=1.5)
ax1.set_ylabel('Teplota [°C]', color='k')
ax1.set_title(f'Simulace teploty - Režim: {CHLAZENI_TYP}\n(Plocha: {area_note})')
ax1.grid(True)
ax1.legend(loc='upper right')

# GRAF 2: MOMENT A VÝKON
ax3.set_xlabel('Čas [s]')
ax3.set_ylabel('Moment [Nm]', color='g')
line1 = ax3.plot(cas_plot, torque_log, 'g-', label='Moment (Nm)')
ax3.tick_params(axis='y', labelcolor='g')
ax3.grid(True)

# Druhá osa pro výkon (kW)
ax4 = ax3.twinx()
ax4.set_ylabel('Výkon [kW]', color='orange')
# Převod výkonu na kW pro graf
power_kw = [p / 1000 for p in power_log]
line2 = ax4.plot(cas_plot, power_kw, color='orange', linestyle='--', label='Výkon (kW)')
ax4.tick_params(axis='y', labelcolor='orange')

# Společná legenda pro druhý graf
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc='upper center')

plt.tight_layout()
plt.show()