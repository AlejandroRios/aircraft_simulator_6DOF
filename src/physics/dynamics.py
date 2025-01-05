"""
dynamics - Função de Dinâmica (para bomba "dumb" com modelo baseado em alpha e beta)
Descrição: Calcula as derivadas dos estados de uma bomba sem controle e sem propulsão,
           mas mantendo o modelo V, alpha, beta, etc.
"""

import numpy as np

# Importações necessárias (fora da função):
from models.Cmat import Cmat
from models.skew import skew  # <-- IMPORTAÇÃO DA FUNÇÃO SKEW AQUI
from physics.aero_loads import aero_loads
from physics.prop_loads import prop_loads
from physics.ISA import ISA
from globals import aircraft, g

def dynamics(t, X, U, W):
    """
    Calcula a dinâmica do sistema (bomba "dumb") mantendo V, alpha, beta, etc.

    Entrada:
    - t (float): tempo atual [s].
    - X (numpy.ndarray): vetor de estados (12 elementos) => [V, alpha, q, theta, H, x, beta, phi, p, r, psi, y]
    - U (numpy.ndarray): vetor de controles (6 elementos) [zerado ou ignorado neste caso].
    - W (numpy.ndarray): vetor de vento (3 elementos) [se não usar, passe zeros].

    Saída:
    - Xdot (numpy.ndarray): derivadas dos estados (12 elementos).
    - Y (numpy.ndarray): variáveis de saída para pós-processamento.
    """

    # ---------------------------------------------------------
    # 1) Extraindo estados
    # ---------------------------------------------------------
    V        = X[0]
    alpha_deg= X[1]
    q_deg_s  = X[2]
    theta_deg= X[3]
    H_m      = X[4]
    x        = X[5]
    beta_deg = X[6]
    phi_deg  = X[7]
    p_deg_s  = X[8]
    r_deg_s  = X[9]
    psi_deg  = X[10]
    y        = X[11]

    # Converte ângulos para radianos
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad  = np.deg2rad(beta_deg)
    phi_rad   = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    psi_rad   = np.deg2rad(psi_deg)

    p_rad_s = np.deg2rad(p_deg_s)
    q_rad_s = np.deg2rad(q_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)

    # ---------------------------------------------------------
    # 2) Cálculo de (u, v, w) no referencial corpo
    # ---------------------------------------------------------
    u = V * np.cos(beta_rad) * np.cos(alpha_rad)
    v = V * np.sin(beta_rad)
    w = V * np.cos(beta_rad) * np.sin(alpha_rad)
    V_b = np.array([u, v, w])  # Velocidade no corpo

    # ---------------------------------------------------------
    # 3) Matrizes de rotação e gravidade
    # ---------------------------------------------------------
    C_phi   = Cmat(1, phi_rad)
    C_theta = Cmat(2, theta_rad)
    C_psi   = Cmat(3, psi_rad)
    # Transformação do veículo para o corpo:
    C_bv    = C_phi @ C_theta @ C_psi

    # Gravidade no referencial corpo
    g_b = C_bv @ np.array([0, 0, g])

    # ---------------------------------------------------------
    # 4) Propriedades físicas (bomba) e matrizes de massa
    # ---------------------------------------------------------
    m     = aircraft['m']
    J_O_b = aircraft['J_O_b']
    rC_b  = aircraft['rC_b']  # vetorzinho de CG

    Mgen = np.block([
        [m * np.eye(3),          -m * skew(rC_b)],
        [m * skew(rC_b),         J_O_b         ]
    ])

    # ---------------------------------------------------------
    # 5) Cargas aerodinâmicas e propulsivas (zeradas)
    # ---------------------------------------------------------
    # Se 'aero_loads' foi simplificada para a bomba, ela deve
    # retornar principalmente arrasto, e possivelmente sustentação pequena.
    Faero_b, Maero_O_b, Yaero = aero_loads(X, U)

    # Zerar a parte de propulsão (sem motor):
    Fprop_b = np.zeros(3)
    Mprop_O_b = np.zeros(3)

    # ---------------------------------------------------------
    # 6) Forças e momentos resultantes
    # ---------------------------------------------------------
    omega_b = np.array([p_rad_s, q_rad_s, r_rad_s])

    eq_F = m * skew(omega_b) @ V_b - m * skew(omega_b) @ skew(rC_b) @ omega_b
    eq_F += Faero_b + Fprop_b + m * g_b  # Fprop_b = 0

    eq_M = skew(omega_b) @ J_O_b @ omega_b
    eq_M += m * skew(rC_b) @ skew(omega_b) @ V_b
    eq_M += Maero_O_b + Mprop_O_b + m * skew(rC_b) @ g_b  # Mprop_O_b=0

    # Resolve o sistema [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    edot = np.linalg.solve(Mgen, np.concatenate([eq_F, eq_M]))
    u_dot, v_dot, w_dot = edot[:3]
    p_dot, q_dot, r_dot = edot[3:6]

    # ---------------------------------------------------------
    # 7) Cinemática angular e translacional
    # ---------------------------------------------------------
    # 7a) Velocidades angulares -> Euler angles
    HPhi_inv = np.column_stack((C_phi[:, 0], C_phi[:, 1], C_bv[:, 2]))
    Phi_dot_rad_s = np.linalg.solve(HPhi_inv, omega_b)

    # 7b) Translação no Eixo Inercial
    dREOdt = C_bv.T @ V_b  # dx/dt, dy/dt, dH/dt

    # ---------------------------------------------------------
    # 8) Derivadas de V, alpha, beta, etc.
    # ---------------------------------------------------------
    # V_dot
    if V > 1e-6:
        V_dot = (V_b @ edot[:3]) / V
    else:
        V_dot = 0.0

    # alpha_dot (em deg/s)
    denom_uw = u**2 + w**2
    if denom_uw > 1e-6:
        alpha_dot_rad_s = (u * w_dot - w * u_dot) / denom_uw
    else:
        alpha_dot_rad_s = 0.0
    alpha_dot_deg_s = np.rad2deg(alpha_dot_rad_s)

    # beta_dot (em deg/s)
    denom_uvw = V * np.sqrt(u**2 + w**2)
    if denom_uvw > 1e-6:
        beta_dot_rad_s = (V * v_dot - v * V_dot) / denom_uvw
    else:
        beta_dot_rad_s = 0.0
    beta_dot_deg_s = np.rad2deg(beta_dot_rad_s)

    # q_dot (em deg/s)
    q_dot_deg_s = np.rad2deg(q_dot)

    # p_dot, r_dot
    p_dot_deg_s = np.rad2deg(p_dot)
    r_dot_deg_s = np.rad2deg(r_dot)

    # theta_dot, phi_dot, psi_dot
    theta_dot_deg = np.rad2deg(Phi_dot_rad_s[1])
    phi_dot_deg   = np.rad2deg(Phi_dot_rad_s[0])
    psi_dot_deg   = np.rad2deg(Phi_dot_rad_s[2])

    # dH, dx, dy
    H_dot = dREOdt[2]
    x_dot = dREOdt[0]
    y_dot = dREOdt[1]

    # ---------------------------------------------------------
    # 9) Monta Xdot
    # ---------------------------------------------------------
    Xdot = np.array([
        V_dot,            # dV/dt
        alpha_dot_deg_s,  # d(alpha)/dt
        q_dot_deg_s,      # dq/dt
        theta_dot_deg,    # d(theta)/dt
        H_dot,            # dH/dt
        x_dot,            # dx/dt
        beta_dot_deg_s,   # d(beta)/dt
        phi_dot_deg,      # d(phi)/dt
        p_dot_deg_s,      # dp/dt
        r_dot_deg_s,      # dr/dt
        psi_dot_deg,      # d(psi)/dt
        y_dot             # dy/dt
    ])

    # ---------------------------------------------------------
    # 10) Variáveis de saída Y
    # ---------------------------------------------------------
    rho, _, _, a = ISA(H_m)
    Mach = V / a if a > 1e-6 else 0.0
    qbar = 0.5 * rho * (V**2)

    n_C_b = -1/(m*g) * (Faero_b + Fprop_b)  # Fprop_b=0
    r_pilot_b = aircraft['r_pilot_b']

    # Aceleração no CG do piloto (se quiser remover, fique à vontade)
    n_pilot_b = n_C_b - (1/g)*(
        skew(edot[3:6]) @ (r_pilot_b - rC_b) +
        skew(omega_b) @ skew(omega_b) @ (r_pilot_b - rC_b)
    )

    Y = np.concatenate([
        [V, alpha_deg, q_deg_s, theta_deg, H_m, x, beta_deg, phi_deg,
         p_deg_s, r_deg_s, psi_deg, y],
        n_pilot_b.flatten(),
        n_C_b.flatten(),
        [Mach, qbar],
        Fprop_b.flatten(),    # = 0
        Mprop_O_b.flatten(),  # = 0
        Yaero.flatten(),      # arrasto, sustentação, etc.
        [V_dot, alpha_dot_deg_s, beta_dot_deg_s, u, v, w]
    ])

    return Xdot, Y
