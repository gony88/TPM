  import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# --- 1. DEFINICE MATERIÁLOVÝCH VLASTNOSTÍ ---

def get_steel_props(T_celsius):
    """
    Vrací vlastnosti uhlíkové oceli v závislosti na teplotě.
    T_celsius: Teplota [°C] (může být číslo nebo pole)
    """
    # Oříznutí teploty (aby vzorce neulétly mimo rozsah 20-1000 °C)
    T = np.clip(T_celsius, 20.0, 1000.0)
    
    # 1. Tepelná vodivost k [W/m.K] (Klesá s teplotou)
    # 20°C -> 52 W/mK, 800°C -> 27 W/mK
    k = 54.0 - 0.028 * T
    
    # 2. Měrná tepelná kapacita cp [J/kg.K] (Roste s teplotou)
    # 20°C -> 450 J/kgK, 800°C -> 800 J/kgK
    c_p = 450.0 + 0.28 * T
    
    # 3. Hustota rho [kg/m3] (Konstantní, změna je zanedbatelná)
    rho = 7850.0
    
    return k, c_p, rho

# --- 2. NAČTENÍ DAT Z EXCELU (MAPA MOTORU) ---

def load_engine_map(filename='motor_data.xlsx'):
    try:
        # Pokus o načtení Excelu
        df = pd.read_excel(filename)
        # Očekáváme sloupce 'RPM' a 'Torque'
        rpm_data = df['RPM'].values
        torque_data = df['Torque'].values
        print(f"ÚSPĚCH: Načten soubor '{filename}'.")
    except FileNotFoundError:
        print(f"CHYBA: Soubor '{filename}' nenalezen!")
        print("POUŽÍVÁM NÁHRADNÍ DATA PRO UKÁZKU (Aby kód nespadl).")
        # Náhradní data jen pro demonstraci, pokud nemáte soubor
        rpm_data = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000])
        torque_data = np.array([0, 800, 1100, 1200, 1150, 900, 700])
    except KeyError:
        print("CHYBA: Excel musí obsahovat sloupce s názvy 'RPM' a 'Torque'.")
        raise

    # Vytvoření interpolační funkce (spojí body z Excelu čarou)
    interp_func = interp1d(rpm_data, torque_data, kind='linear', fill_value="extrapolate")
    return interp_func, rpm_data, torque_data

# Inicializace mapy motoru
get_torque_from_rpm, map_rpm, map_torque = load_engine_map()

# --- 3. GEOMETRIE A KONSTANTY ---

# Referenční hodnoty pro Betu
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)
rho_f = 2500.0; c_f = 1000.0; k_f = 0.2

# Beta
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)
print(f"Koeficient Beta: {beta:.3f}")

# Geometrie
r_out = 0.124 #Vnější poloměr [m]
r_in = 0.0875 #Vnitřní poloměr [m]
tloustka_oceli = 0.004 #Tloušťka ocelové lamely [m]

S_friction = np.pi * (r_out**2 - r_in**2)
n_pairs = 14 

# --- 4. ZATĚŽOVACÍ CYKLUS ---

# Definice otáček (Skluz) - Otáčky se mění v čase lineárně (brzdění/rozběh)
n_start_slip = 1000  # [rpm] Otáčky na začátku
n_end_slip = 0.0       # [rpm] Otáčky na konci

n_cyklu = 2
t_zab = 0.5
t_pauza = 5.0
t_cyklus = t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# --- 5. SIMULACE ---

L = tloustka_oceli / 2 
N = 50                 
dx = L / (N - 1)       

# Stabilita
k_c, c_c, rho_c = get_steel_props(20.0)
dt = 0.9 * (0.5 * dx**2 / (k_c / (rho_c * c_c)))

# Inicializace
T = np.full(N, 70.0)
T_new = np.copy(T)

# Pole pro ukládání dat do grafů
cas_plot = []
T_surf_plot = []
T_core_plot = []

# Pro graf vlastností
cp_surf_plot = [] 
k_surf_plot = []

t = 0.0
step = 0

print(f"\nSpouštím simulaci...")

while t < t_total:
    t_local = t % t_cyklus
    
    # A. VÝPOČET ZDROJE TEPLA
    if t_local <= t_zab:
        ratio = t_local / t_zab
        
        # 1. Zjistíme aktuální otáčky
        current_rpm = n_start_slip + (n_end_slip - n_start_slip) * ratio
        
        # 2. Vytáhneme moment z EXCELOVÉ křivky
        real_torque = get_torque_from_rpm(current_rpm)
        if real_torque < 0: real_torque = 0
        
        # 3. Výkon = Moment * Omega
        omega = current_rpm * 2 * np.pi / 60
        power = real_torque * omega
        
        # 4. Tepelný tok
        q_net = (power / n_pairs / S_friction) * beta
    else:
        q_net = 0.0

    # B. AKTUALIZACE VLASTNOSTÍ (PRO SOLVER I GRAFY)
    # Získáme vlastnosti pro celé teplotní pole
    k_vec, cp_vec, rho_val = get_steel_props(T)
    alpha_vec = k_vec / (rho_val * cp_vec)

    # C. SOLVER
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
    T_new[0] = T[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
    T_new[-1] = T[-1] + alpha_vec[-1] * dt / dx**2 * (T[-2] - T[-1])

    T[:] = T_new[:]
    t += dt
    step += 1

    # Ukládání dat
    if step % 100 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])
        
        # Ukládáme vlastnosti na POVRCHU (kde je teplota nejvyšší a změny největší)
        k_surf, cp_surf, _ = get_steel_props(T[0])
        k_surf_plot.append(k_surf)
        cp_surf_plot.append(cp_surf)

# --- 6. VYKRESLENÍ GRAFŮ (Dle zadání) ---

fig = plt.figure(figsize=(12, 12))

# GRAF 1: Teploty (Povrch vs Střed)
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(cas_plot, T_surf_plot, 'r-', label='Povrch (x=0)', linewidth=1.5)
ax1.plot(cas_plot, T_core_plot, 'b--', label='Střed (x=tl/2)', linewidth=1.5)
ax1.set_ylabel('Teplota [°C]')
ax1.set_title('Průběh teploty na lamele')
ax1.legend(loc='upper right')
ax1.grid(True)

# GRAF 2: Vlastnosti materiálu (Kapacita a Vodivost)
ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
color = 'tab:red'
ax2.set_ylabel('Měrná tepelná kapacita Cp [J/kg.K]', color=color)
ax2.plot(cas_plot, cp_surf_plot, color=color, linestyle='-', label='Cp (Kapacita)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(True)

# Druhá osa Y pro vodivost
ax2_twin = ax2.twinx()
color = 'tab:blue'
ax2_twin.set_ylabel('Tepelná vodivost k [W/m.K]', color=color)
ax2_twin.plot(cas_plot, k_surf_plot, color=color, linestyle='--', label='k (Vodivost)')
ax2_twin.tick_params(axis='y', labelcolor=color)
ax2_twin.set_title('Změna vlastností materiálu na povrchu (Závislosti)')

# GRAF 3: Načtená mapovací křivka (Kontrola Excelu)
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(map_rpm, map_torque, 'k-o', linewidth=2, markersize=4)
ax3.set_xlabel('Otáčky [RPM]')
ax3.set_ylabel('Točivý moment [Nm]')
ax3.set_title('Použitá charakteristika motoru (z Excelu)')
ax3.grid(True)
# Zvýrazníme oblast, ve které jsme se pohybovali v simulaci
ax3.axvspan(min(n_end_slip, n_start_slip), max(n_end_slip, n_start_slip), color='green', alpha=0.1, label='Pracovní rozsah simulace')
ax3.legend()

plt.tight_layout()
plt.show()
