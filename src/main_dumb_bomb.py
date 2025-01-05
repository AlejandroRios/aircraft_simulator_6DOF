"""
Simulador 6DOF de Bomba Dumb
Descrição: Análise de voo balístico sem controle de direção ou propulsão
Autor: Seu Nome
Data de Criação: YYYY-MM-DD
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# 1. Definição de constantes e parâmetros
# =============================================================================
g = 9.80665  # Aceleração da gravidade (m/s²)

# Se quiser considerar forma e coeficientes aerodinâmicos (opcional):
Cd = 0.2     # Coeficiente de arrasto
A_ref = 0.05 # Área de referência [m^2]
rho = 1.225  # Densidade do ar ao nível do mar (kg/m^3) - simplificado
m = 100.0    # Massa da bomba (kg) - exemplo
# Inércias (caso queira simular rotação 6DOF de fato):
Ixx = 10.0
Iyy = 10.0
Izz = 5.0
J = np.diag([Ixx, Iyy, Izz])

# =============================================================================
# 2. Equações de movimento (função dynamics)
# =============================================================================
def dynamics_bomba(t, X):
    """
    Equações 6DOF simplificadas:
       X = [
         u, v, w,       (velocidades no eixo do corpo)
         p, q, r,       (velocidades angulares do corpo)
         xE, yE, zE,    (posições inerciais ou ângulos Euler, dependendo do modelo)
         phi, theta, psi (ângulos de rolamento, arfagem, guinada) -- se desejado
       ]
    Caso queira algo ainda mais simples (somente translação), reduza o tamanho de X.
    """

    # Para ilustrar, vou considerar somente 6 estados de translação (3 posições + 3 velocidades).
    # Se você quer 12 estados (com rotações e velocidades angulares), adapte aqui.
    # X = [ x,  y,  z,  Vx, Vy, Vz ] (6 estados)
    xE, yE, zE, Vx, Vy, Vz = X

    # Cálculo de velocidade e arrasto (opcional)
    V_mod = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    # Força de arrasto (simplicada)
    F_drag = 0.5 * rho * (V_mod**2) * Cd * A_ref

    # Direção do arrasto é contrária à velocidade
    if V_mod > 1e-6:
        ax_drag = -F_drag * (Vx / V_mod) / m
        ay_drag = -F_drag * (Vy / V_mod) / m
        az_drag = -F_drag * (Vz / V_mod) / m
    else:
        # Se V=0 (evitar divisão por zero)
        ax_drag, ay_drag, az_drag = (0.0, 0.0, 0.0)
    
    # Aceleração total
    ax = ax_drag       # + outras forças se houver
    ay = ay_drag
    az = -g + az_drag  # gravidade + arrasto em z

    # Equações do modelo translacional
    dxE = Vx
    dyE = Vy
    dzE = Vz
    dVx = ax
    dVy = ay
    dVz = az

    # Retorne o vetor de derivadas
    return [dxE, dyE, dzE, dVx, dVy, dVz]


# =============================================================================
# 3. Condições iniciais de simulação
# =============================================================================
# Exemplo: bomba solta a 3000 m de altitude, com velocidade inicial horizontal
x0 = 0.0      # posição inicial Eixo X (m)
y0 = 0.0      # posição inicial Eixo Y (m)
z0 = 3000.0   # altitude inicial (m)
Vx0 = 200.0   # velocidade inicial no eixo X (m/s)
Vy0 = 0.0     # velocidade inicial no eixo Y (m/s)
Vz0 = 0.0     # velocidade inicial vertical (m/s) (se for para baixo, use negativo)

X0 = [x0, y0, z0, Vx0, Vy0, Vz0]

# Parâmetros de tempo
t0 = 0.0     # tempo inicial
tf = 60.0    # tempo final de simulação (s)
dt = 0.01    # passo de avaliação (s)
t_eval = np.arange(t0, tf + dt, dt)

# =============================================================================
# 4. Integração numérica
# =============================================================================
sol = solve_ivp(
    fun=dynamics_bomba,
    t_span=(t0, tf),
    y0=X0,
    method='RK45',
    t_eval=t_eval,
    max_step=dt
)

# Extraindo resultados
Tsol = sol.t
Xsol = sol.y.T  # cada linha de Xsol é [x, y, z, Vx, Vy, Vz] num tempo Tsol[i]

# =============================================================================
# 5. Análise e plotagem
# =============================================================================
# Exemplo simples de plot: trajetória X vs Z
fig, ax = plt.subplots(figsize=(7,5))

ax.plot(Xsol[:,0], Xsol[:,2], label="Trajetória (x vs. z)")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("Trajetória da Bomba Dumb (sem controle nem propulsão)")
ax.grid(True)
ax.legend()

# Se quiser ver a evolução de z(t):
fig2, ax2 = plt.subplots()
ax2.plot(Tsol, Xsol[:,2], label="z(t)")
ax2.invert_yaxis()  # para mostrar altitude "cima" no gráfico
ax2.set_xlabel("Tempo (s)")
ax2.set_ylabel("Altitude (m)")
ax2.set_title("Perfil de altitude ao longo do tempo")
ax2.grid(True)
ax2.legend()

plt.show()
