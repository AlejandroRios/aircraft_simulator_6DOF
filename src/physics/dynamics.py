"""
dynamics - Função de Dinâmica da Aeronave
Descrição: Calcula as derivadas dos estados da aeronave e as variáveis de saída.
Entrada:
- t (float): Tempo atual (s).
- X (numpy.ndarray): Vetor de estados (12 elementos).
- U (numpy.ndarray): Vetor de controles (6 elementos).
- W (numpy.ndarray): Vetor de vento (3 elementos).
Saída:
- Xdot (numpy.ndarray): Derivadas dos estados (12 elementos).
- Y (numpy.ndarray): Variáveis de saída da dinâmica.
"""

import numpy as np
from models.Cmat import Cmat
from models.skew import skew
from physics.aero_loads import aero_loads
from physics.prop_loads import prop_loads
from physics.ISA import ISA
from globals import aircraft, g


def dynamics(t, X, U, W):
    """
    Calcula a dinâmica da aeronave.

    Parâmetros:
    - t: float, tempo atual (s).
    - X: numpy.ndarray, vetor de estados (12 elementos).
    - U: numpy.ndarray, vetor de controles (6 elementos).
    - W: numpy.ndarray, vetor de vento (3 elementos).

    Retorno:
    - Xdot: numpy.ndarray, derivadas dos estados (12 elementos).
    - Y: numpy.ndarray, variáveis de saída da dinâmica.
    """

    # Estados extraídos de X
    V = X[0]
    alpha_deg = X[1]
    q_deg_s = X[2]
    theta_deg = X[3]
    H_m = X[4]
    x = X[5]
    beta_deg = X[6]
    phi_deg = X[7]
    p_deg_s = X[8]
    r_deg_s = X[9]
    psi_deg = X[10]
    y = X[11]

    # Conversões para radianos
    alpha_rad = np.deg2rad(alpha_deg)
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    p_rad_s = np.deg2rad(p_deg_s)
    q_rad_s = np.deg2rad(q_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)
    psi_rad = np.deg2rad(psi_deg)
    beta_rad = np.deg2rad(beta_deg)

    # Velocidades no referencial corpo
    u = V * np.cos(beta_rad) * np.cos(alpha_rad)
    v = V * np.sin(beta_rad)
    w = V * np.cos(beta_rad) * np.sin(alpha_rad)
    V_b = np.array([u, v, w])

    # Matrizes de rotação
    C_phi = Cmat(1, phi_rad)
    C_theta = Cmat(2, theta_rad)
    C_psi = Cmat(3, psi_rad)
    C_bv = C_phi @ C_theta @ C_psi

    # Aceleração gravitacional no referencial corpo
    g_b = C_bv @ np.array([0, 0, g])

    # Propriedades da aeronave
    m = aircraft['m']
    J_O_b = aircraft['J_O_b']
    rC_b = aircraft['rC_b']

    # Matrizes de massa generalizada
    Mgen = np.block([
        [m * np.eye(3), -m * skew(rC_b)],
        [m * skew(rC_b), J_O_b]
    ])

    # Cargas aerodinâmicas e propulsivas
    Faero_b, Maero_O_b, _ = aero_loads(X, U)
    Fprop_b, Mprop_O_b, _ = prop_loads(X, U)

    # Equações de forças e momentos
    omega_b = np.array([p_rad_s, q_rad_s, r_rad_s])
    eq_F = m * skew(omega_b) @ V_b - m * skew(omega_b) @ skew(rC_b) @ omega_b
    eq_F += Faero_b + Fprop_b + m * g_b

    eq_M = skew(omega_b) @ J_O_b @ omega_b + m * skew(rC_b) @ skew(omega_b) @ V_b
    eq_M += Maero_O_b + Mprop_O_b + m * skew(rC_b) @ g_b

    # Resolução do sistema de equações
    edot = np.linalg.solve(Mgen, np.concatenate([eq_F, eq_M]))
    u_dot, v_dot, w_dot = edot[:3]

    # Cinemática angular
    HPhi_inv = np.column_stack((C_phi[:, 0], C_phi[:, 1], C_bv[:, 2]))
    Phi_dot_rad_s = np.linalg.solve(HPhi_inv, omega_b)

    # Cinemática translacional
    dREOdt = C_bv.T @ V_b

    # Atualização de estados
    V_dot = (V_b @ edot[:3]) / V
    alpha_dot_deg_s = np.rad2deg((u * w_dot - w * u_dot) / (u**2 + w**2))
    q_dot_deg_s = np.rad2deg(edot[4])
    theta_dot_deg = np.rad2deg(Phi_dot_rad_s[1])
    H_dot = dREOdt[2]
    x_dot = dREOdt[0]
    beta_dot_deg_s = np.rad2deg((V * v_dot - v * V_dot) / (V * np.sqrt(u**2 + w**2)))
    phi_dot_deg = np.rad2deg(Phi_dot_rad_s[0])
    p_dot_deg_s = np.rad2deg(edot[3])
    r_dot_deg_s = np.rad2deg(edot[5])
    psi_dot_deg = np.rad2deg(Phi_dot_rad_s[2])
    y_dot = dREOdt[1]

    # Vetor de estados derivados
    Xdot = np.array([
        V_dot, alpha_dot_deg_s, q_dot_deg_s, theta_dot_deg,
        H_dot, x_dot, beta_dot_deg_s, phi_dot_deg,
        p_dot_deg_s, r_dot_deg_s, psi_dot_deg, y_dot
    ])

    # Variáveis de saída
    rho, _, _, a = ISA(H_m)
    Mach = V / a
    qbar = 0.5 * rho * V**2
    n_C_b = -1 / (m * g) * (Faero_b + Fprop_b)

    r_pilot_b = aircraft['r_pilot_b']
    n_pilot_b = n_C_b - (1 / g) * (
        skew(edot[3:6]) @ (r_pilot_b - rC_b) +
        skew(omega_b) @ skew(omega_b) @ (r_pilot_b - rC_b)
    )

    Y = np.concatenate([
        X, n_pilot_b, n_C_b, [Mach, qbar], Fprop_b, Mprop_O_b
    ])

    return Xdot, Y
