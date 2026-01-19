import numpy as np
import matplotlib.pyplot as plt

# --- 1. DEFINICE MATERIÁLOVÝCH VLASTNOSTÍ (Funkce) ---

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

# --- 2. KONSTANTY PRO VÝPOČET BETA A GEOMETRIE ---

# Referenční hodnoty pro ocel (pouze pro výpočet Bety na začátku)
# Pro samotný výpočet teploty se už nepoužijí, tam jede funkce výše.
k_s_ref, c_s_ref, rho_s_ref = get_steel_props(70.0)

# Materiál: Třecí obložení (Konstanty)
rho_f = 2500.0
c_f = 1000.0
k_f = 0.2

# Výpočet Beta (Statický odhad rozdělení tepla)
b_steel = np.sqrt(k_s_ref * rho_s_ref * c_s_ref)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

print(f"Koeficient Beta (podíl tepla do oceli): {beta:.3f}")

# --- 3. GEOMETRIE A VÝKON ---

# Rozměry spojky
r_out = 0.124         # [m]
r_in = 0.101          # [m]
tloustka_oceli = 0.005 # [m]

# Počáteční teplota
T_start = 70.0        # [°C]

# Výpočet PLOCHY
S_celkova = np.pi * (r_out**2 - r_in**2)
S_friction = S_celkova  # 100% plochy

print(f"Celková plocha (S_friction): {S_friction*10000:.2f} cm2")

# Výkonové zatížení
P_mot_peak = 293000.0 # [W] 
n_pairs = 14          # Počet třecích kontaktů

# Tok tepla
q_nominal_peak = (P_mot_peak / n_pairs) / S_friction
q_max_input = q_nominal_peak * beta

print(f"Špičkový tok do oceli: {q_max_input/1e6:.2f} MW/m2")

# --- 4. SIMULACE (FDM 1D s proměnnými vlastnostmi) ---

# Parametry cyklu
n_cyklu = 3
t_zab = 1.0
t_pauza = 8.0
t_cyklus = t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Diskretizace
L = tloustka_oceli / 2 
N = 50                 
dx = L / (N - 1)       

# Výpočet časového kroku dt (Stabilita)
# Musíme použít "nejrychlejší" difuzivitu (ta je při nízké teplotě)
# alpha = k / (rho * cp). Za studena je k nejvyšší a cp nejnižší -> alpha je MAX.
k_cold, c_cold, rho_cold = get_steel_props(20.0)
alpha_max = k_cold / (rho_cold * c_cold)
dt = 0.9 * (0.5 * dx**2 / alpha_max)

print(f"Časový krok dt: {dt:.6f} s")

# Inicializace polí
T = np.full(N, T_start)
T_new = np.copy(T)

# Pro grafy
cas_plot = []
T_surf_plot = []
T_core_plot = []
k_surf_plot = [] # Pro zajímavost budeme sledovat, jak padá vodivost

t = 0.0
step = 0

print("\nSpouštím simulaci s nelineárními vlastnostmi oceli...")

while t < t_total:
    t_local = t % t_cyklus
    
    # 1. Okrajová podmínka (Zdroj tepla)
    if t_local <= t_zab:
        q_net = q_max_input * (1 - t_local / t_zab)
    else:
        q_net = 0.0

    # 2. AKTUALIZACE VLASTNOSTÍ MATERIÁLU
    # Pro aktuální teplotní pole T získáme pole vlastností
    k_vec, cp_vec, rho_val = get_steel_props(T)
    
    # Vypočteme alpha pro každý uzel zvlášť (vektor)
    alpha_vec = k_vec / (rho_val * cp_vec)

    # 3. SOLVER (FDM)
    
    # Vnitřní uzly (Heat Equation s proměnným alpha)
    # T_new[i] = T[i] + alpha[i] * ...
    T_new[1:-1] = T[1:-1] + alpha_vec[1:-1] * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])

    # Povrch (x=0) - Neumannova podmínka
    # Používáme vlastnosti povrchu (k_vec[0], cp_vec[0])
    T_new[0] = T[0] + (dt / (rho_val * cp_vec[0] * (dx/2))) * (q_net - k_vec[0] * (T[0] - T[1]) / dx)
    
    # Střed (x=L) - Adiabatická
    # Používáme vlastnosti středu (alpha_vec[-1])
    T_new[-1] = T[-1] + alpha_vec[-1] * dt / dx**2 * (T[-2] - T[-1])

    # Update
    T[:] = T_new[:]
    t += dt
    step += 1

    if step % 200 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])
        k_surf_plot.append(k_vec[0]) # Ukládáme vodivost na povrchu

# --- 5. VYKRESLENÍ ---

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Graf Teploty
ax1.plot(cas_plot, T_surf_plot, 'r-', label='Povrch (x=0)', linewidth=1.5)
ax1.plot(cas_plot, T_core_plot, 'b--', label='Střed (x=tl/2)', linewidth=1.5)
ax1.set_ylabel('Teplota [°C]')
ax1.set_title(f'Simulace s proměnnými vlastnostmi oceli (k, Cp = f(T))')
ax1.legend()
ax1.grid(True)

# Graf Vodivosti (pro kontrolu, že to funguje)
ax2.plot(cas_plot, k_surf_plot, 'g-', label='Tepelná vodivost na povrchu', linewidth=1.5)
ax2.set_ylabel('Vodivost k [W/m.K]')
ax2.set_xlabel('Čas [s]')
ax2.set_title('Změna tepelné vodivosti oceli během ohřevu')
ax2.legend()
ax2.grid(True)
ax2.invert_yaxis() # Aby bylo vidět, jak klesá (volitelné)

plt.tight_layout()
plt.show()
