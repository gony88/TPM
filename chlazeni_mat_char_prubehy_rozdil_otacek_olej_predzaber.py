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
# Pokud nerozumíte rovnicím dole, stačí měnit hodnoty jen v této sekci.

# --- A. NASTAVENÍ CHLAZENÍ ---
# Jakým způsobem má model počítat odvod tepla (součinitel 'h')?
# MOŽNOSTI:
#   "STATIC"     = Použije fixní hodnotu h (jako kdyby se motor netočil nebo točil pomalu). 
#                  Vhodné pro konzervativní odhad.
#   "ANALYTIC"   = (DOPORUČENO) Dynamicky počítá h podle otáček motoru (odstředivá síla), 
#                  geometrie drážek a viskozity oleje. Nejvěrnější realitě.
#   "FLOW_LIMIT" = Ignoruje hydrodynamiku a počítá maximální možné chlazení 
#                  dané tepelnou kapacitou průtoku oleje (Q = m * c * dT).
#   "NO_COOLING" = Vypne chlazení (h = 0). Simuluje selhání čerpadla nebo "nejhorší možný scénář".

CHLAZENI_TYP = "NO_COOLING"   # <--- ZDE ZMĚŇTE TYP CHLAZENÍ

# Máme započítat, že drážky snižují stykovou plochu?
# True  = Výkon jde do menší plochy (pouze vrcholky drážek) -> Větší tepelný tok q -> Vyšší teplota.
# False = Výkon se rozpočítá na celou plochu mezikruží (jako by byla hladká).
INCLUDE_AREA_REDUCTION = False  # <--- ZDE ZMĚŇTE VOLBU

# --- B. NOVÉ: NASTAVENÍ ROZJEZDU DO KOPCE (HILL START) ---
# Tato sekce umožňuje simulovat fázi "předzáběru", kdy auto stojí v kopci,
# motor běží v otáčkách a spojka prokluzuje konstantním momentem, aby auto necouvlo.

ENABLE_HILL_START = True   # Zapnout simulaci kopce? (True/False)

# Parametry pro HILL START (Použijí se jen když ENABLE_HILL_START = True)
t_hold       = 1.0    # [s]   Doba držení auta na spojce (předzáběr) před samotným rozjezdem.
torque_hold  = 1200.0 # [Nm]  Statický moment potřebný k udržení auta ve svahu.
n_motor_hold = 1800.0 # [rpm] Otáčky motoru, které řidič drží při stání v kopci.

# --- C. TEPLOTY OLEJE A LIMITY ---
# Simulujeme stav s VÝMĚNÍKEM TEPLA (chladičem), který udržuje vstupní olej na konstantní teplotě.

T_olej_inlet = 70.0   # [°C] Teplota čerstvého oleje, který vstupuje do hřídele (z chladiče).
T_film_max   = 150.0  # [°C] Maximální teplota olejového filmu pro výpočet viskozity.
                      # Fyzikální význam: I když má ocel 300°C, proudící olej se nestihne ohřát
                      # na více než cca 150°C, než vyteče ven. Zabraňuje to výpočtu 
                      # nesmyslně nízké viskozity (jako voda).

# --- D. PRŮBĚH OTÁČEK (SAMOTNÝ ROZJEZD - LAUNCH) ---
# Toto nastává AŽ PO skončení fáze držení v kopci (nebo hned, pokud kopec není).
# Rozlišujeme, co dělá motor (pohání čerpadlo) a co dělá spojka (prokluzuje).

# 1. OTÁČKY MOTORU (Absolutní)
# Ovlivňují: Moment motoru (z mapy) a Chlazení (odstředivá síla).
n_motor_start = 2000.0  # [rpm] Otáčky motoru na začátku pohybu vozidla
n_motor_end   = 2000.0  # [rpm] Otáčky motoru na konci prokluzu (synchronizace)
n_motor_idle  = 800.0   # [rpm] Otáčky motoru v PAUZE (volnoběh)

# 2. OTÁČKY PROKLUZU (Relativní rozdíl)
# Ovlivňují: Generované teplo (Výkon = Moment * Skluz).
# Musí jít z maxima (n_start) do nuly (n_end = sepnuto).
n_slip_start  = 2000.0  # [rpm] Počáteční prokluz (rozdíl rychlostí)
n_slip_end    = 0.0     # [rpm] Konec prokluzu (vždy 0)

# --- E. ČASOVÁNÍ CYKLU ---
n_cyklu = 1           # Počet opakování rozjezdu
t_zab   = 1.74        # [s] Doba trvání dynamického prokluzu (samotný rozjezd)
t_pauza = 5.0         # [s] Doba chladnutí mezi rozjezdy

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

def get_cooling_analytical(rpm, T_film_actual, geometry):
    """
    POKROČILÝ VÝPOČET SOUČINITELE PŘESTUPU TEPLA 'h'
    ------------------------------------------------
    Tato funkce simuluje chování oleje v rotujících drážkách (Waffle pattern).
    Funguje jako odstředivé čerpadlo.
    
    Vstupy:
      rpm           : Aktuální otáčky koše spojky (motoru)
      T_film_actual : Teplota oleje přímo v drážce (určuje viskozitu)
      geometry      : Slovník s rozměry (poloměry, hydraulický průměr)
    """
    # Ošetření nulových otáček (vždy je tam alespoň malá konvekce)
    if rpm < 10: return 50.0 
    
    # A. Vlastnosti oleje (ATF - Automatic Transmission Fluid)
    rho = 850.0       # Hustota [kg/m3]
    lam_oil = 0.14    # Tepelná vodivost oleje [W/m.K]
    c_oil = 2000.0    # Tepelná kapacita oleje [J/kg.K]
    
    # Viskozita (JAK MOC OLEJ TEČE) - Klíčový parametr!
    # Interpolujeme kinematickou viskozitu 'nu' podle teploty filmu.
    # 40°C -> 30 cSt (hustý), 100°C -> 7 cSt (řídký)
    nu = np.interp(T_film_actual, [40, 100], [30e-6, 7e-6]) 
    mu = nu * rho     # Dynamická viskozita [Pa.s]
    
    # Prandtl number (poměr viskozity a tepelné difuzivity)
    Pr = (mu * c_oil) / lam_oil

    # B. Rozbalení geometrie
    r_in, r_out = geometry['r_in'], geometry['r_out']
    Dh = geometry['Dh']     # Hydraulický průměr drážky (jak je "velká" pro průtok)
    L = r_out - r_in        # Délka drážky (cesta oleje)
    
    # C. Odstředivý tlak (Driving Force)
    # Tlak, který vzniká rotací a tlačí olej ven. Roste s druhou mocninou otáček!
    omega = rpm * (2 * np.pi / 60)
    dP = 0.5 * rho * omega**2 * (r_out**2 - r_in**2)
    
    # D. Rychlost proudění v drážce (v_oil)
    # Zjednodušená rovnice pro průtok kanálkem (Poiseuille / Bernoulli)
    # K_shape = 24.0 je konstanta pro obdélníkový průřez drážky
    K_shape = 24.0
    v_oil = (dP * Dh**2) / (K_shape * mu * L)
    
    # E. Reynoldsovo číslo (Re)
    # Určuje, zda je tok klidný (laminární) nebo divoký (turbulentní).
    # Vyšší Re = Lepší chlazení.
    Re = (v_oil * Dh) / nu
    
    # F. Nusseltovo číslo (Nu) - Bezrozměrná intenzita chlazení
    if Re < 2300:
        # Laminární oblast (Pomalé otáčky nebo hustý olej)
        # Sieder-Tate korelace (zohledňuje náběh toku)
        Nu = max(1.86 * (Re * Pr * (Dh/L))**(1/3), 3.66) 
    else:
        # Turbulentní oblast (Vysoké otáčky nebo horký olej)
        # Dittus-Boelter rovnice
        Nu = 0.023 * Re**0.8 * Pr**0.4
        
    # G. Výpočet finálního 'h'
    h = (Nu * lam_oil) / Dh
    
    # Waffle Factor: Zvýšení přestupu tepla díky "rozbíjení" proudu o mřížku
    h *= 1.5 
    
    return h

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
# 1. Statické h (fixní - laminární)
Nu_stat = 6.05
h_static_val = (Nu_stat * lambda_oil) / Dh

# 2. Flow Limit h (limit průtoku)
q_total_lmin = 6.0 # L/min
mdot_per_surface = (q_total_lmin / 60 / 1000 * 850) / n_pairs 
h_flow_limit_val = (mdot_per_surface * c_oil) / S_cooling

# 3. Analytické h (Demo hodnota pro startovní otáčky)
# Použijeme funkci, kterou máme, abychom uživateli ukázali, kolik to bude na začátku
h_analytic_demo_val = get_cooling_analytical(n_motor_start, T_olej_inlet, geometry_dict)

# Výpis info do konzole - PŘEHLEDNÁ TABULKA
print("-" * 60)
print(f"INFO: NASTAVENÍ SIMULACE")
print(f"  * Režim chlazení (ZVOLENÝ): {CHLAZENI_TYP}")
print(f"  * Rozjezd do kopce: {'AKTIVNÍ' if ENABLE_HILL_START else 'NEAKTIVNÍ'}")
if ENABLE_HILL_START:
    print(f"    -> Doba držení (Hold): {t_hold} s")
    print(f"    -> Moment držení:      {torque_hold} Nm")
print(f"  * Teplota přívodu oleje: {T_olej_inlet} °C")
print(f"  * Max teplota filmu (limit): {T_film_max} °C")
print("-" * 60)
print("PŘEHLED MOŽNÝCH HODNOT CHLAZENÍ (PRO INFO A POROVNÁNÍ):")
print(f"  1. STATIC (Laminární):     {h_static_val:.1f} W/m2K")
print(f"  2. FLOW LIMIT (Kapacita):  {h_flow_limit_val:.1f} W/m2K (při {q_total_lmin} L/min)")
print(f"  3. ANALYTIC (při {n_motor_start:.0f} rpm): {h_analytic_demo_val:.1f} W/m2K (Dynamicky se mění)")
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
torque_log = []     # Moment
power_log = []      # Výkon
T_oil_film_log = [] # Teplota olejového filmu

# Proměnné pro hledání maxim
max_torque_rec = 0.0; max_power_rec = 0.0; max_q_net_rec = 0.0

t = 0.0
step = 0

print("... Simulace běží ...")

while t < t_total:
    t_local = t % t_cyklus # Lokální čas v rámci jednoho cyklu (0 až t_cyklus)

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
        real_torque = torque_hold  # Fixní moment pro udržení ve svahu
        omega_slip = rpm_slip * 2 * np.pi / 60
        power = real_torque * omega_slip
        
        # Chlazení: Olej teče jen drážkami
        cooling_ratio = ratio_groove 

    # --- FÁZE 2: DYNAMICKÝ ROZJEZD (LAUNCH) ---
    elif t_local < (t_hold + t_zab):
        # Auto se rozjíždí, prokluz klesá
        
        # Musíme posunout čas, aby rampa začala od 0 (relativně k začátku rozjezdu)
        t_in_launch = t_local - t_hold 
        ratio = t_in_launch / t_zab
        
        # Otáčky
        rpm_engine_abs = n_motor_start + (n_motor_end - n_motor_start) * ratio
        rpm_slip = n_slip_start * (1 - ratio)
        
        # Moment (z mapy motoru) a Výkon
        real_torque = get_torque_from_rpm(rpm_engine_abs)
        if real_torque < 0: real_torque = 0
        
        omega_slip = rpm_slip * 2 * np.pi / 60
        power = real_torque * omega_slip
        
        # Chlazení: Olej teče jen drážkami
        cooling_ratio = ratio_groove

    # --- FÁZE 3: PAUZA (COOLING) ---
    else:
        # Spojka rozepnuta nebo plně sepnuta bez prokluzu
        
        rpm_engine_abs = n_motor_idle # Motor běží na volnoběh
        rpm_slip = 0.0                # Žádný prokluz = žádné tření
        
        real_torque = 0.0
        power = 0.0
        
        # Chlazení: Stejný režim (drážky), ale zdroj tepla je 0
        cooling_ratio = ratio_groove

    # ==========================================================
    # VÝPOČET CHLAZENÍ (h)
    # ==========================================================
    if CHLAZENI_TYP == "STATIC":
        h_current = h_static_val
        T_film_used = T_olej_inlet # Jen pro graf
        
    elif CHLAZENI_TYP == "FLOW_LIMIT":
        h_current = h_flow_limit_val
        T_film_used = T_olej_inlet
        
    elif CHLAZENI_TYP == "NO_COOLING":
        h_current = 0.0
        T_film_used = T_olej_inlet
        
    else: # REŽIM "ANALYTIC" (Nejpřesnější)
        # 1. Spočítáme teoretickou teplotu filmu (průměr mezi horkým povrchem a studeným olejem)
        T_film_calc = (T[0] + T_olej_inlet) / 2 
        
        # 2. Aplikujeme LIMIT (zastropování)
        # Protože olej proudí rychle, nestihne se ohřát na extrémní teploty.
        if T_film_calc > T_film_max:
            T_film_used = T_film_max
        else:
            T_film_used = T_film_calc
            
        # 3. Zavoláme funkci pro výpočet h (s použitím otáček motoru a limitované teploty)
        h_current = get_cooling_analytical(rpm_engine_abs, T_film_used, geometry_dict)


    # ==========================================================
    # TEPELNÁ BILANCE (q_net)
    # ==========================================================
    
    # 1. Vstup tepla (Generation)
    q_gen = (power / n_pairs / S_calc_power) * beta
    
    # 2. Odvod tepla (Cooling)
    q_cool = h_current * (T[0] - T_olej_inlet) * cooling_ratio
    
    # 3. Výsledek
    q_net = q_gen - q_cool
    
    # Uložení maxim pro statistiku
    if real_torque > max_torque_rec: max_torque_rec = real_torque
    if power > max_power_rec: max_power_rec = power
    if q_net > max_q_net_rec: max_q_net_rec = q_net

    # ==========================================================
    # SOLVER (FDM)
    # ==========================================================
    # Aktualizace materiálových vlastností pro aktuální teplotu (nelinearita)
    k_vec, cp_vec, rho_val = get_steel_props(T)
    alpha_vec = k_vec / (rho_val * cp_vec) # Teplotní difuzivita

    # A. Vnitřní uzly (vedení tepla uvnitř materiálu)
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
 
    # B. Povrchový uzel (zde vstupuje q_net)
    T_new[0] = T[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
   
    # C. Středový uzel (symetrie, adiabatická stěna)
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
        power_log.append(power)
        T_oil_film_log.append(T_film_used)

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
print(f"Špičkový výkon (celkový):    {max_power_rec / 1000:.1f} kW")
print(f"Špičkový tepelný tok (q):    {max_q_net_rec / 1e6:.2f} MW/m²")
print(f"MAXIMÁLNÍ TEPLOTA POVRCHU:   {max(T_surf_plot):.1f} °C")
print("="*60)

# Vytvoření okna se 4 GRAFY pod sebou
# figsize zvětšena na (12, 16), aby se 4 grafy vešly
fig, (ax1, ax3, ax5, ax7) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# --- GRAF 1: TEPLOTA MATERIÁLU ---
ax1.plot(cas_plot, T_surf_plot, 'r-', label='Povrch Oceli (Styk s obložením)', linewidth=1.5)
ax1.plot(cas_plot, T_core_plot, 'b--', label='Střed Oceli (Symetrie)', linewidth=1.5)
ax1.set_ylabel('Teplota [°C]', fontsize=12, fontweight='bold')
ax1.set_title(f'1. Průběh teploty ocelové lamely\n(Režim: {CHLAZENI_TYP}, Vstup oleje: {T_olej_inlet}°C)', fontsize=14)
ax1.grid(True, alpha=0.5)
ax1.legend(loc='upper right', fontsize=10)

# --- GRAF 2: ZÁTĚŽ (MOMENT A VÝKON) ---
ax3.set_ylabel('Moment [Nm]', color='green', fontsize=12, fontweight='bold')
line1 = ax3.plot(cas_plot, torque_log, 'g-', label='Moment motoru (Nm)', alpha=0.8)
ax3.tick_params(axis='y', labelcolor='green')
ax3.grid(True, alpha=0.5)

# Druhá osa Y pro výkon
ax4 = ax3.twinx()
ax4.set_ylabel('Tepelný Výkon [kW]', color='orange', fontsize=12, fontweight='bold')
power_kw = [p / 1000 for p in power_log]
line2 = ax4.plot(cas_plot, power_kw, color='orange', linestyle='--', label='Třecí Výkon (kW)')
ax4.tick_params(axis='y', labelcolor='orange')

# Společná legenda
lns = line1 + line2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc='center right', fontsize=10)
ax3.set_title('2. Zatížení spojky v čase (Zdroj tepla)', fontsize=12)

# --- GRAF 3: STAV OLEJE (TEPLOTA FILMU) ---
ax5.set_ylabel('Teplota Filmu [°C]', color='purple', fontsize=12, fontweight='bold')
ax5.plot(cas_plot, T_oil_film_log, 'm-', label='Efektivní teplota filmu (pro viskozitu)', linewidth=2)
ax5.axhline(y=T_olej_inlet, color='gray', linestyle='--', label=f'Teplota Přívodu ({T_olej_inlet}°C)')
ax5.axhline(y=T_film_max, color='red', linestyle=':', label=f'Limit ({T_film_max}°C)')
ax5.fill_between(cas_plot, T_olej_inlet, T_oil_film_log, color='purple', alpha=0.1) # Vybarvení
ax5.set_title('3. Teplota oleje v drážce (Ovlivňuje viskozitu)', fontsize=12)
ax5.grid(True, alpha=0.5)
ax5.legend(loc='upper right', fontsize=10)

# --- NOVÉ: GRAF 4 - SOUČINITEL PŘESTUPU TEPLA (h) ---
ax7.set_xlabel('Čas simulace [s]', fontsize=12, fontweight='bold')
ax7.set_ylabel('h [W/m2K]', color='blue', fontsize=12, fontweight='bold')
ax7.plot(cas_plot, h_log, 'c-', label='Součinitel přestupu tepla h', linewidth=2)
ax7.fill_between(cas_plot, 0, h_log, color='cyan', alpha=0.1) # Vybarvení pod křivkou
ax7.set_title('4. Intenzita chlazení (Průběh h v čase)', fontsize=12)
ax7.grid(True, alpha=0.5)
ax7.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()