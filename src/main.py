"""
Simulador 6DOF - Main para bomba MK 82
Descrição: Configuração inicial e simulação balística de uma bomba Mk 82 sem controle.
Autor: Seu Nome
Data de Criação: YYYY-MM-DD
"""

import numpy as np
from scipy.integrate import solve_ivp

# Se sua função dynamics estiver no módulo physics/dynamics.py, mantenha a importação:
from physics.dynamics import dynamics

# Plotagem final (assumindo que você tenha essa função):
from plot_all_final import plot_all_final

# Se você tiver um arquivo globals.py que define g ou algo similar:
from globals import g

# ============================================================================
# 1. Constantes globais e conversões
# ============================================================================
m2ft = 1 / 0.3048
ft2m = 1 / m2ft
lb2kg = 0.45359237
kg2lb = 1 / lb2kg
slug2kg = g * lb2kg
kg2slug = 1 / slug2kg
deg2rad = np.pi / 180
rad2deg = 1 / deg2rad

# ============================================================================
# 2. Dados aproximados da bomba MK 82
# ============================================================================
m_bomba = 227.0  # Massa (kg)
diametro = 0.273  # m
S_bomba = np.pi * (diametro / 2)**2  # ~0,0585 m²

# Momentos de inércia (aprox.)
Ixx = 3.0     
Iyy = 75.0    
Izz = 75.0
Ixy = 0.0
Ixz = 0.0
Iyz = 0.0

J_bomba = np.array([
    [Ixx, -Ixy, -Ixz],
    [-Ixy, Iyy, -Iyz],
    [-Ixz, -Iyz, Izz]
])

# ============================================================================
# 3. Dicionário "aircraft" representando a bomba
# ============================================================================
aircraft = {
    'm': m_bomba,
    'J_O_b': J_bomba,
    'rC_b': np.array([0.0, 0.0, 0.0]),
    'b': diametro,  # “envergadura” ~ diâmetro
    'S': S_bomba,   
    'c': diametro,  # “corda” ~ diâmetro
    'hex': 0.0,     
    'r_pilot_b': np.array([0.0, 0.0, 0.0])
}

# ============================================================================
# 4. Condições iniciais (sem trim)
# ============================================================================
H_inicial = 3000.0  # m (altitude)
V_inicial = 200.0   # m/s (horizontal)

# Estado 6DOF (12 estados): [u, v, w, p, q, r, x, y, z, phi, theta, psi]
# Mas aqui, vamos definir:
X0 = np.zeros(12)
X0[0] = V_inicial  # u = 200 m/s no corpo
X0[8] = 0.0        # z = 0 no inercial (veremos uso)
X0[4] = H_inicial  # Vamos usar a posição X[4] como altitude
# O resto fica em zero

# Sem superfícies de controle (bomba “dumb”), então U e vento = zeros
U_bomba = np.zeros(6)
W_bomba = np.zeros(3)

# ============================================================================
# 5. Tempo de simulação
# ============================================================================
tf = 2.0
dt = 0.01
t_span = (0, tf)
t_eval = np.arange(0, tf + dt, dt)

# ============================================================================
# 6. Integração
# ============================================================================
sol = solve_ivp(
    fun=lambda t, X: dynamics(t, X, U_bomba, W_bomba)[0],
    t_span=t_span,
    y0=X0,
    method='RK45',
    t_eval=t_eval,
    max_step=dt
)

Tsol = sol.t
Xsol = sol.y.T  # (n_points, 12)

# Reconstruindo Ysol para cada tempo
Ysol = []
for i, t in enumerate(Tsol):
    _, Y_i = dynamics(t, Xsol[i], U_bomba, W_bomba)
    Ysol.append(Y_i)
Ysol = np.array(Ysol)

# Monta Usol p/ plot
Usol = np.tile(U_bomba, (len(Tsol), 1))

# ============================================================================
# 7. Plot
# ============================================================================
plot_all_final(Tsol, Ysol, Usol, Xsol)

print("Simulação da bomba Mk 82 concluída.")
