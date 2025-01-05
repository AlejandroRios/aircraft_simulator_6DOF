"""
Simulador 6DOF - Main para bomba MK 82
Descrição: Configuração inicial e simulação balística de uma bomba Mk 82 sem controle e sem trimagem.
Autor: Seu Nome
Data de Criação: YYYY-MM-DD
"""

import numpy as np
from scipy.integrate import solve_ivp

# Se sua função dynamics estiver no módulo physics/dynamics.py, mantenha a importação:
from physics.dynamics import dynamics

# Se quiser usar ISA para altitude/velocidade do som, etc. (opcional):
from physics.ISA import ISA

# Plotagem final (assumindo que você tenha esse script):
from plot_all_final import plot_all_final

# Se você tiver um arquivo globals.py que define g ou algo similar:
from globals import g

# =============================================================================
# 1. Constantes globais e conversões
# =============================================================================
m2ft = 1 / 0.3048
ft2m = 1 / m2ft
lb2kg = 0.45359237
kg2lb = 1 / lb2kg
slug2kg = g * lb2kg
kg2slug = 1 / slug2kg
deg2rad = np.pi / 180
rad2deg = 1 / deg2rad

# =============================================================================
# 2. Dados aproximados da bomba MK 82
# =============================================================================
m_bomba = 227.0  # Massa (kg), ~500 lb
# Estimando área frontal:
diametro = 0.273  # m
S_bomba = np.pi * (diametro / 2)**2  # ~0,0585 m²

# Momentos de inércia (aprox.):
Ixx = 3.0     # kg·m² (eixo longitudinal, pode variar)
Iyy = 75.0    # kg·m² (eixo transversal)
Izz = 75.0    # kg·m² (eixo transversal)
Ixy = 0.0
Ixz = 0.0
Iyz = 0.0

J_bomba = np.array([
    [Ixx, -Ixy, -Ixz],
    [-Ixy, Iyy, -Iyz],
    [-Ixz, -Iyz, Izz]
])

# =============================================================================
# 3. Montagem do dicionário "aircraft" (aqui, nossa 'bomba')
# =============================================================================
# Você pode manter a mesma estrutura do seu projeto para reutilizar dynamics, etc.
aircraft = {
    'm': m_bomba,     # Massa
    'J_O_b': J_bomba, # Tensor de inércia
    'rC_b': np.array([0.0, 0.0, 0.0]),  # CG no referencial de corpo (pode ser 0,0,0)
    'b': diametro,    # Se quiser associar 'envergadura' ~ diâmetro (não é muito relevante para bomba)
    'S': S_bomba,     
    'c': diametro,    # 'corda' ~ diâmetro, novamente irrelevante mas mantido para compatibilidade
    'hex': 0.0,       # Sem motor, sem torque do eixo
    'r_pilot_b': np.array([0.0, 0.0, 0.0])  # Sem piloto, mas mantido se dynamics usa
}

# =============================================================================
# 4. Condições iniciais (sem trim)
# =============================================================================
# Exemplo: Lançar a 3000 m de altitude, velocidade ~200 m/s horizontal
H_inicial = 3000.0  # m
V_inicial = 200.0   # m/s (horizontal)
z0 = H_inicial

# Em um modelo 6DOF (12 estados), por exemplo:
# [u, v, w, p, q, r, x, y, z, phi, theta, psi]
X0 = np.zeros(12)
X0[0] = V_inicial  # u = 200 m/s
X0[4] = z0         # z = 3000 m
# Se quiser algum pequeno ângulo de ataque ou rotação inicial, ajuste aqui.

# Sem superfícies de controle (bomba “dumb”), então U ~ zeros
U_bomba = np.zeros(6)  # Ajuste se seu dynamics esperar outro dimensionamento
W_bomba = np.zeros(3)  # Se seu dynamics usar vento ou alguma força externa, senão zero

# =============================================================================
# 5. Tempo de simulação
# =============================================================================
tf = 5.0     # tempo final (s)
dt = 0.01     # passo para solver
t_span = (0, tf)
t_eval = np.arange(0, tf + dt, dt)

# =============================================================================
# 6. Integração numérica das EDOs
# =============================================================================
sol = solve_ivp(
    fun=lambda t, X: dynamics(t, X, U_bomba, W_bomba)[0],
    t_span=t_span,
    y0=X0,
    method='RK45',
    t_eval=t_eval,
    max_step=dt
)

# Extraindo resultado
Tsol = sol.t
Xsol = sol.y.T  # Transpõe para que cada linha seja um instante de tempo

# Se a dinâmica retorna também Y (saídas), podemos calcular a cada passo:
Ysol = []
for i, t in enumerate(Tsol):
    _, Y_i = dynamics(t, Xsol[i], U_bomba, W_bomba)
    Ysol.append(Y_i)
Ysol = np.array(Ysol)

# Montando Usol para consistência com a função de plot
Usol = np.tile(U_bomba, (len(Tsol), 1))

# =============================================================================
# 7. Plotagem dos resultados
# =============================================================================
plot_all_final(Tsol, Ysol, Usol, Xsol)

print("Simulação da bomba Mk 82 concluída. Verifique os gráficos para analisar a trajetória.")
