import numpy as np
import matplotlib.pyplot as plt

# --- 1. DEFINICE MATERIÁLOVÝCH VLASTNOSTÍ ---

# Fixní tepelná kapacita (změna dle zadání)
c_p_ocel = 490.0  # [J/kg.K] - Průměrná hodnota pro ocel
rho_ocel = 7850.0 # [kg/m3]  - Hustota

def get_steel_k(T_celsius):
    """
    Vrací pouze tepelnou vodivost k [W/m.K], která se mění s teplotou.
    """
    T = np.clip(T_celsius, 20.0, 1000.0)
    
    # Tepelná vodivost k [W/m.K] (Klesá s teplotou)
    # 20°C -> 53.4 W/mK, vyšší teplota -> nižší vodivost
    k = 54.0 - 0.028 * T
    
    return k

# --- 2. KONSTANTY PRO VÝPOČET BETA A GEOMETRIE ---

# Referenční vodivost pro výpočet Bety (při 70°C)
k_s_ref = get_steel_k(70.0)

# Materiál: Třecí obložení (Konstanty)
rho_f = 2500.0
c_f = 1000.0
k_f = 0.2

# Výpočet Beta (Statický odhad rozdělení tepla)
# Nyní používáme fixní c_p_ocel
b_steel = np.sqrt(k_s_ref * rho_ocel * c_p_ocel)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

print(f"Koeficient Beta (podíl tepla do oceli): {beta:.3f}")

# --- 3. GEOMETRIE A VÝKON ---

# Rozměry spojky
r_out = 0.124         # [m]
r_in = 0.0875         # [m]
tloustka_oceli = 0.002 # [m]

# Počáteční teplota
T_start = 80.0        # [°C]

# Výpočet PLOCHY
S_celkova = np.pi * (r_out**2 - r_in**2)
S_friction = S_celkova  # 100% plochy

print(f"Celková plocha (S_friction): {S_friction*10000:.2f} cm2")

# Výkonové zatížení
P_mot_peak = 300000 # [W] 
n_pairs = 14          # Počet třecích kontaktů

# Tok tepla
q_nominal_peak = (P_mot_peak / n_pairs) / S_friction
q_max_input = q_nominal_peak * beta

print(f"Špičkový tok do oceli: {q_max_input/1e6:.2f} MW/m2")

# --- 4. SIMULACE (FDM 1D) ---

# Parametry cyklu
n_cyklu = 1
t_zab = 2.0
t_pauza = 1.0
t_cyklus = t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Diskretizace
L = tloustka_oceli / 2 
N = 50                 
dx = L / (N - 1)       

# Výpočet časového kroku dt (Stabilita)
# alpha = k / (rho * cp). 
# Použijeme maximální možné k (při 20°C) pro nejhorší případ stability.
k_max = get_steel_k(20.0)
alpha_max = k_max / (rho_ocel * c_p_ocel)
dt = 0.9 * (0.5 * dx**2 / alpha_max)

print(f"Časový krok dt: {dt:.6f} s")

# Inicializace polí
T = np.full(N, T_start)
T_new = np.copy(T)

# Pro grafy
cas_plot = []
T_surf_plot = []
T_core_plot = []

t = 0.0
step = 0

print("\nSpouštím simulaci (k=proměnné, cp=konstantní)...")

while t < t_total:
    t_local = t % t_cyklus
    
    # 1. Okrajová podmínka (Zdroj tepla - klesající rampa)
    if t_local <= t_zab:
        q_net = q_max_input * (1 - t_local / t_zab)
    else:
        q_net = 0.0

    # 2. AKTUALIZACE VLASTNOSTÍ MATERIÁLU
    # Získáme pouze vodivost k v závislosti na T
    k_vec = get_steel_k(T)
    
    # Vypočteme alpha (teplotní difuzivitu)
    # k se mění, rho a cp jsou fixní konstanty
    alpha_vec = k_vec / (rho_ocel * c_p_ocel)

    # 3. SOLVER (FDM)
    
    # Vnitřní uzly
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])

    # Povrch (x=0)
    T_new[0] = T[0] + (dt / (rho_ocel * c_p_ocel * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
    
    # Střed (x=L)
    T_new[-1] = T[-1] + alpha_vec[-1] * dt / dx**2 * (T[-2] - T[-1])

    # Update
    T[:] = T_new[:]
    t += dt
    step += 1

    if step % 200 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])

# --- 5. VYKRESLENÍ (Pouze Teplota) ---

plt.figure(figsize=(10, 6))
plt.plot(cas_plot, T_surf_plot, 'r-', label='Povrch (x=0)', linewidth=1.5)
plt.plot(cas_plot, T_core_plot, 'b--', label='Střed (x=tl/2)', linewidth=1.5)

plt.xlabel('Čas [s]')
plt.ylabel('Teplota [°C]')
plt.title(f'Simulace teploty (k=f(T), Cp={c_p_ocel} J/kgK)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()