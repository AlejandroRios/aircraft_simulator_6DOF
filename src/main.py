"""
Simulador de Aeronaves 6DOF - Main
Descrição: Configuração inicial, cálculo das condições de equilíbrio e simulação.
Autor: Seu Nome
Data de Criação: YYYY-MM-DD
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

from models.trimGNBA import trimGNBA
from models.state_vec import state_vec
from models.control_vec import control_vec
from physics.dynamics import dynamics
from physics.ISA import ISA
from plot_all_final import plot_all_final
from globals import aircraft, g

# Constantes globais

m2ft = 1 / 0.3048
ft2m = 1 / m2ft
lb2kg = 0.45359237
kg2lb = 1 / lb2kg
slug2kg = g * lb2kg
kg2slug = 1 / slug2kg
deg2rad = np.pi / 180
rad2deg = 1 / deg2rad

# Dados geométricos e de inércia
b = 32.757  # Envergadura (m)
S = 116  # Área de referência (m²)
c = 3.862  # Corda média aerodinâmica (m)
W = 55788 * g  # Peso (N)
m = W / g  # Massa (kg)

Ixx = 821466
Iyy = 3343669
Izz = 4056813
Ixy = 0
Ixz = 178919
Iyz = 0

J = np.array([
    [Ixx, -Ixy, -Ixz],
    [-Ixy, Iyy, -Iyz],
    [-Ixz, -Iyz, Izz]
])

# Configuração de CG e outras propriedades
xCG, yCG, zCG = 0, 0, 0
rC_b = np.array([xCG, yCG, zCG])
r_pilot_b = np.array([15, 0, 0])
hex = 160

aircraft = {
    'm': m,
    'J_O_b': J,
    'rC_b': rC_b,
    'b': b,
    'S': S,
    'c': c,
    'hex': hex,
    'r_pilot_b': r_pilot_b
}

# Condições de equilíbrio
H_m_eq = 38000 * ft2m
rho, _, _, a = ISA(H_m_eq)

Mach = 0.78
V_eq = Mach * a

trim_par = {
    'V': V_eq,
    'H_m': H_m_eq,
    'chi_deg': 0,
    'gamma_deg': 0,
    'phi_dot_deg_s': 0,
    'theta_dot_deg_s': 0,
    'psi_dot_deg_s': 0,
    'beta_deg_eq': 0,
    'W': np.zeros(3)
}

# Resolver a condição de trim
x_eq_0 = np.zeros(15)
x_eq_0[0] = V_eq

x_eq = fsolve(trimGNBA, x_eq_0, args=(trim_par,))
X_eq = state_vec(x_eq, trim_par)
U_eq = control_vec(x_eq)
Xdot_eq, Y_eq = dynamics(0, X_eq, U_eq, trim_par['W'])

# Resultados
print("----- PARÂMETROS DE VOO TRIMADOS -----")
print(f"x_CG = {xCG:.4f} m")
print(f"y_CG = {yCG:.4f} m")
print(f"z_CG = {zCG:.4f} m")
print(f"gamma = {trim_par['gamma_deg']:.4f} deg")
print(f"V = {X_eq[0]:.2f} m/s")
print(f"alpha = {X_eq[2]:.4f} deg")
print(f"beta = {X_eq[7]:.4f} deg")
print(f"Mach = {Y_eq[19]:.4f}")

# Simulação
tf = 50 if trim_par['psi_dot_deg_s'] == 0 else 360 / trim_par['psi_dot_deg_s']
dt = 1e-5

sol = solve_ivp(
    lambda t, X: dynamics(t, X, U_eq, trim_par['W'])[0],  # Use apenas Xdot
    [0, tf],
    X_eq,
    t_eval=np.arange(0, tf, dt),
    max_step=dt
)

# Extraia as informações de tempo e solução
Tsol = sol.t  # Valores do tempo
Xsol = sol.y.T  # Solução transposta para alinhar com a convenção usada

# Ysol e Usol devem ser calculados manualmente após a integração
Usol = np.tile(U_eq, (len(Tsol), 1))  # Replique U_eq ao longo de Tsol
Ysol = np.array([dynamics(t, Xsol[i], U_eq, trim_par['W'])[1] for i, t in enumerate(Tsol)])
print(f"Ysol shape: {Ysol.shape}")

# Chame a função de plotagem
plot_all_final(Tsol, Ysol, Usol, Xsol)