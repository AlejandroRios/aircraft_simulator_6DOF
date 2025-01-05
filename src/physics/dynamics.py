"""
dynamics - Dinâmica (bomba "dumb" 6DOF em graus para alpha/beta)
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
    Estados (12):
    [V, alpha_deg, q_deg_s, theta_deg,  H (altitude),   x, beta_deg, phi_deg,
     p_deg_s,       r_deg_s,  psi_deg,  y ]
    OU
    [u, v, w, p, q, r, x, y, z, phi, theta, psi] dependendo do design.
    
    Aqui, assumimos: [V, alpha, q, theta, H, x, beta, phi, p, r, psi, y].
    """

    # ------------------------------------------------------------------
    # 1) Extrair estados
    # ------------------------------------------------------------------
    V         = X[0]
    alpha_deg = X[1]
    q_deg_s   = X[2]
    theta_deg = X[3]
    H_m       = X[4]
    xE        = X[5]
    beta_deg  = X[6]
    phi_deg   = X[7]
    p_deg_s   = X[8]
    r_deg_s   = X[9]
    psi_deg   = X[10]
    yE        = X[11]

    # Converte alguns ângulos e velocidades
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad  = np.deg2rad(beta_deg)
    phi_rad   = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    psi_rad   = np.deg2rad(psi_deg)

    p_rad_s = np.deg2rad(p_deg_s)
    q_rad_s = np.deg2rad(q_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)

    # ------------------------------------------------------------------
    # 2) Montar (u,v,w) no corpo a partir de V, alpha, beta
    # ------------------------------------------------------------------
    u = V * np.cos(beta_rad) * np.cos(alpha_rad)
    v = V * np.sin(beta_rad)
    w = V * np.cos(beta_rad) * np.sin(alpha_rad)
    V_b = np.array([u, v, w])  # Velocidade no corpo

    # ------------------------------------------------------------------
    # 3) Matrizes de rotação e gravidade
    # ------------------------------------------------------------------
    # Define rotação de inercial -> corpo (Z para baixo no corpo).
    # Se a aeronave está sem ângulo, o body z aponta p/ baixo.
    C_phi   = Cmat(1, phi_rad)
    C_theta = Cmat(2, theta_rad)
    C_psi   = Cmat(3, psi_rad)
    # C_bv: transf. do VEÍCULO (inercial) p/ BODY
    C_bv = C_phi @ C_theta @ C_psi

    # Gravidade no body. Se z_inercial é p/ cima, e z_body é p/ baixo,
    # então g deve ser negative no body se pitch=roll=yaw=0.
    # Mais simples: aplique [0,0,-g] em inercial, depois transforme p/ body:
    g_inertial = np.array([0, 0, -g])
    g_b = C_bv @ g_inertial  # gravidade no corpo

    # ------------------------------------------------------------------
    # 4) Propriedades físicas & matrizes de massa
    # ------------------------------------------------------------------
    m     = aircraft['m']
    J_O_b = aircraft['J_O_b']
    rC_b  = aircraft['rC_b']

    Mgen = np.block([
        [m * np.eye(3),          -m * skew(rC_b)],
        [m * skew(rC_b),         J_O_b]
    ])

    # ------------------------------------------------------------------
    # 5) Aero & Prop (para “bomba dumb,” prop=0)
    # ------------------------------------------------------------------
    Faero_b, Maero_O_b, Yaero = aero_loads(X, U)
    Fprop_b = np.zeros(3)
    Mprop_O_b = np.zeros(3)

    # ------------------------------------------------------------------
    # 6) Somar forças e momentos no corpo
    # ------------------------------------------------------------------
    omega_b = np.array([p_rad_s, q_rad_s, r_rad_s])
    eq_F = (m * skew(omega_b) @ V_b 
            - m * skew(omega_b) @ skew(rC_b) @ omega_b
            + Faero_b + Fprop_b
            + m * g_b
           )

    eq_M = (skew(omega_b) @ J_O_b @ omega_b
            + m * skew(rC_b) @ skew(omega_b) @ V_b
            + Maero_O_b + Mprop_O_b
            + m * skew(rC_b) @ g_b
           )

    edot = np.linalg.solve(Mgen, np.concatenate([eq_F, eq_M]))
    u_dot, v_dot, w_dot = edot[:3]
    p_dot, q_dot, r_dot = edot[3:6]

    # ------------------------------------------------------------------
    # 7) Cinemática: Euler angles & posição inercial
    # ------------------------------------------------------------------
    # a) Euler angles
    #   Precisamos descobrir d(phi)/dt, d(theta)/dt, d(psi)/dt
    #   Em seu código original, você usa:
    HPhi_inv = np.column_stack((C_phi[:, 0], C_phi[:, 1], C_bv[:, 2]))
    Phi_dot_rad_s = np.linalg.solve(HPhi_inv, omega_b)

    # b) Posição inercial
    #   dREOdt = [x_dotE, y_dotE, z_dotE] = C_bv^T * V_b
    #   Lembre: se z_inercial é p/ cima, e w>0 for "para baixo" no corpo,
    #   então z_dotE tende a ser negativo => altitude cai.
    dREOdt = C_bv.T @ V_b

    # ------------------------------------------------------------------
    # 8) Derivadas de V, alpha, beta (em deg/s etc.)
    # ------------------------------------------------------------------
    V_mod = np.linalg.norm(V_b)
    if V_mod > 1e-6:
        V_dot = (V_b @ edot[:3]) / V_mod
    else:
        V_dot = 0.0

    # alpha_dot
    denom_uw = u**2 + w**2
    if denom_uw > 1e-6:
        alpha_dot_rad_s = (u*w_dot - w*u_dot)/denom_uw
    else:
        alpha_dot_rad_s = 0.0
    alpha_dot_deg_s = np.rad2deg(alpha_dot_rad_s)

    # beta_dot
    denom_uvw = V_mod*np.sqrt(u**2 + w**2)
    if denom_uvw > 1e-6:
        beta_dot_rad_s = (V_mod*v_dot - v*V_dot)/denom_uvw
    else:
        beta_dot_rad_s = 0.0
    beta_dot_deg_s = np.rad2deg(beta_dot_rad_s)

    # q_dot (p, r idem) => deg/s
    p_dot_deg_s = np.rad2deg(p_dot)
    q_dot_deg_s = np.rad2deg(q_dot)
    r_dot_deg_s = np.rad2deg(r_dot)

    # Euler angles => deg
    phi_dot_deg   = np.rad2deg(Phi_dot_rad_s[0])
    theta_dot_deg = np.rad2deg(Phi_dot_rad_s[1])
    psi_dot_deg   = np.rad2deg(Phi_dot_rad_s[2])

    # ------------------------------------------------------------------
    # 9) Atualização de altitude e coords
    # ------------------------------------------------------------------
    # Se dREOdt[2] é a taxa de variação de z_inercial (positivo "para cima").
    # Se a altitude H = - z_inercial (exemplo), precisamos H_dot = - dREOdt[2].
    # Para que, se z_inercial decresce (bomba descendo), altitude aumenta?
    # Na verdade, altitude deve cair => H_dot = - z_dot => se z_dot é negativo.
    H_dot = - dREOdt[2]   # altitude cai se z_dotE for negativo
    x_dot = dREOdt[0]     # xE
    y_dot = dREOdt[1]     # yE

    # ------------------------------------------------------------------
    # 10) Monta Xdot
    # ------------------------------------------------------------------
    Xdot = np.array([
        V_dot,            # dV/dt
        alpha_dot_deg_s,  # d(alpha)/dt
        q_dot_deg_s,      # d(q)/dt
        theta_dot_deg,    # d(theta)/dt
        H_dot,            # d(altitude)/dt
        x_dot,            # dxE/dt
        beta_dot_deg_s,   # d(beta)/dt
        phi_dot_deg,      # d(phi)/dt
        p_dot_deg_s,      # d(p)/dt
        r_dot_deg_s,      # d(r)/dt
        psi_dot_deg,      # d(psi)/dt
        y_dot             # dyE/dt
    ])

    # ------------------------------------------------------------------
    # 11) Variáveis de saída p/ debugging
    # ------------------------------------------------------------------
    rho, _, _, a_loc = ISA(H_m)
    Mach = V / a_loc if a_loc>1e-6 else 0.0
    qbar = 0.5*rho*(V**2)

    n_C_b = -1/(m*g) * (Faero_b + Fprop_b)  # normalizado
    r_pilot_b = aircraft['r_pilot_b']
    n_pilot_b = n_C_b - (1/g)*(
        skew(edot[3:6]) @ (r_pilot_b - rC_b) +
        skew(omega_b) @ skew(omega_b) @ (r_pilot_b - rC_b)
    )

    Y = np.concatenate([
        [V, alpha_deg, q_deg_s, theta_deg, H_m, xE, beta_deg, phi_deg,
         p_deg_s, r_deg_s, psi_deg, yE],
        n_pilot_b.flatten(),
        n_C_b.flatten(),
        [Mach, qbar],
        Fprop_b.flatten(),
        Mprop_O_b.flatten(),
        Yaero.flatten(),
        [V_dot, alpha_dot_deg_s, beta_dot_deg_s, u, v, w]
    ])

    return Xdot, Y
