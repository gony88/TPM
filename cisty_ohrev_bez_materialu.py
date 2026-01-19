import numpy as np
import matplotlib.pyplot as plt

# --- 1. FYZIKÁLNÍ KONSTANTY ---

# Materiál: Ocel (Separační lamela)
rho_s = 7850.0       # [kg/m3]
c_s = 460.0          # [J/kg.K]
k_s = 52.0           # [W/m.K]
alpha = k_s / (rho_s * c_s)

# Materiál: Třecí obložení (jen pro výpočet rozdělení tepla Beta)
rho_f = 2500.0
c_f = 1000.0
k_f = 0.2

# Výpočet Beta (kolik % tepla jde do oceli a kolik do obložení)
b_steel = np.sqrt(k_s * rho_s * c_s)
b_fric = np.sqrt(k_f * rho_f * c_f)
beta = b_steel / (b_steel + b_fric)

print(f"Koeficient Beta (podíl tepla do oceli): {beta:.3f}")

# --- 2. GEOMETRIE A VÝKON (BEZ DRÁŽEK) ---

# Rozměry spojky
r_out = 0.124         # [m]
r_in = 0.0875          # [m]
tloustka_oceli = 0.003 # [m]

# Počáteční teplota
T_start = 70.0        # [°C]

# Výpočet PLOCHY (Nyní čisté mezikruží bez odečtu drážek)
S_celkova = np.pi * (r_out**2 - r_in**2)
S_friction = S_celkova  # Třecí plocha je 100% plochy

print(f"Celková plocha (S_friction): {S_friction*10000:.2f} cm2")

# Výkonové zatížení
P_mot_peak = 293000.0 # [W] Celkový výkon motoru
n_pairs = 14          # Počet třecích kontaktů

# Tok tepla na jeden kontakt (W/m2)
# Rozdělíme výkon na počet ploch a podělíme celou plochou mezikruží
q_nominal_peak = (P_mot_peak / n_pairs) / S_friction

# Aplikujeme Betu (část tepla, co jde do oceli)
q_max_input = q_nominal_peak * beta

print(f"Špičkový tok do oceli: {q_max_input/1e6:.2f} MW/m2")


# --- 3. SIMULACE (FDM 1D - ČISTÝ OHŘEV) ---

# Parametry cyklu
n_cyklu = 4        # Počet cyklů
t_zab = 1.74       # Čas záběru [s]
t_pauza =   38.26    # Čas pauzy [s]
t_cyklus = t_zab + t_pauza
t_total = n_cyklu * t_cyklus

# Diskretizace sítě
L = tloustka_oceli / 2  # Poloviční tloušťka (symetrie)
N = 50                  # Počet uzlů
dx = L / (N - 1)        # Vzdálenost mezi uzly
dt = 0.9 * (0.5 * dx**2 / alpha) # Časový krok dle stability

# Inicializace polí
T = np.full(N, T_start)
T_new = np.copy(T)

# Pro grafy
cas_plot = []
T_surf_plot = []
T_core_plot = []

t = 0.0
step = 0

print("\nSpouštím simulaci čistého ohřevu...")

while t < t_total:
    t_local = t % t_cyklus
    
    # --- OKRAJOVÉ PODMÍNKY ---
    
    if t_local <= t_zab:
        # Fáze ZÁBĚRU: Lineárně klesající výkon
        # q(t) = q_max * (1 - t / t_zab)
        q_net = q_max_input * (1 - t_local / t_zab)
    else:
        # Fáze PAUZY: Žádný zdroj tepla, žádné chlazení
        # Dokonalá izolace -> teplo se jen rozlije do hmoty (vyrovnání teplot)
        q_net = 0.0

    # --- VÝPOČET TEPLOT (Solver) ---

    # 1. Vnitřní uzly (Vedení tepla uvnitř materiálu)
    T_new[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])

    # 2. Povrch (x=0): Zde vstupuje q_net
    # Rovnice: dT/dt * konst = q_net - vedení_dovnitř
    T_new[0] = T[0] + (dt / (rho_s * c_s * (dx/2))) * (q_net - k_s * (T[0] - T[1]) / dx)
    
    # 3. Střed (x=L): Adiabatická podmínka (symetrie)
    # Teplo neteče dál, odráží se zpět
    T_new[-1] = T[-1] + alpha * dt / dx**2 * (T[-2] - T[-1])

    # Update
    T[:] = T_new[:]
    t += dt
    step += 1

    # Ukládání dat pro graf (ne každý krok, aby to nebylo obří)
    if step % 200 == 0:
        cas_plot.append(t)
        T_surf_plot.append(T[0])
        T_core_plot.append(T[-1])

# --- 4. VYKRESLENÍ ---

plt.figure(figsize=(10, 6))
plt.plot(cas_plot, T_surf_plot, 'r-', label='Povrch (x=0)', linewidth=1.5)
plt.plot(cas_plot, T_core_plot, 'b--', label='Střed (x=tl/2)', linewidth=1.5)

plt.title(f'Čistý ohřev bez chlazení a bez vlivu drážek\n(Plocha={S_friction*10000:.1f} cm2, Max Flux={q_max_input/1e6:.2f} MW/m2)')
plt.xlabel('Čas [s]')
plt.ylabel('Teplota [°C]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
