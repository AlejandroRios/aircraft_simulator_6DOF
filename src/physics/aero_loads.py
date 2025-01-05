"""
aero_loads - Cálculo das Cargas Aerodinâmicas em eixo de corpo (body axis)
com coeficientes DATCOM, incluindo derivações p*b/(2V), q*c/(2V), etc.
Trabalhando em graus (alpha_deg, beta_deg).
"""

import numpy as np
from physics.ISA import ISA
from globals import aircraft, g

def aero_loads(X, U):
    """
    Retorna Faero_b e Maero_O_b em [N] e [N·m].
    Eixo de corpo DATCOM: x_b p/ frente, y_b p/ direita, z_b p/ baixo.
    Se CN>0 => força "para cima" => Z_b = -CN * qS.
    """

    # 1) Extrair estados (todos em GRAUS p/ alpha, beta, p, q, r)
    V         = X[0]
    alpha_deg = X[1]
    q_deg_s   = X[2]
    H_m       = X[4]
    beta_deg  = X[6]
    p_deg_s   = X[8]
    r_deg_s   = X[9]

    # Precisamos de p, q, r em rad/s p/ escalar p*b/(2V), etc.
    p_rad_s = np.deg2rad(p_deg_s)
    q_rad_s = np.deg2rad(q_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)

    # alpha e beta em graus p/ se somar a derivações do DATCOM
    # (Se precisar de trig, converta p/ rad local).
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad  = np.deg2rad(beta_deg)

    # 2) Propriedades do ar e geometria
    rho, _, _, a = ISA(H_m)
    q_bar = 0.5 * rho * (V**2)
    Mach = V/a if a>1e-6 else 0.0

    m  = aircraft['m']
    S  = aircraft['S']
    b  = aircraft['b']
    c  = aircraft['c']

    # 3) Parâmetros adimensionais
    pb_2V = (p_rad_s*b)/(2*V) if V>1e-6 else 0.0
    qc_2V = (q_rad_s*c)/(2*V) if V>1e-6 else 0.0
    rb_2V = (r_rad_s*b)/(2*V) if V>1e-6 else 0.0

    # 4) EXEMPLO DE coeficientes DATCOM (fictícios)
    #    Ajuste conforme seu caso real
    CN_0     =  0.2
    CN_alpha =  0.03  # (1/deg)
    CN_q     =  3.2
    CN_beta  =  0.01
    CN_p     =  0.1
    CN_r     =  0.05

    CN = ( CN_0
           + CN_alpha*alpha_deg
           + CN_q*qc_2V
           + CN_beta*beta_deg
           + CN_p*pb_2V
           + CN_r*rb_2V
         )

    CA_0       = 0.05
    CA_alpha   = 0.002
    CA_q       = 0.0
    CA_beta    = 0.0
    CA_p       = 0.0
    CA_r       = 0.0

    CA = ( CA_0
           + CA_alpha*alpha_deg
           + CA_q*qc_2V
           + CA_beta*beta_deg
           + CA_p*pb_2V
           + CA_r*rb_2V
         )

    CY_0    = 0.0
    CY_beta = 0.02
    CY_p    = -0.2
    CY_r    = 0.3

    CY = ( CY_0
           + CY_beta*beta_deg
           + CY_p*pb_2V
           + CY_r*rb_2V
         )

    # Momentos
    Cl_0     = 0.0
    Cl_beta  = 0.00
    Cl_p     = -0.40
    Cl_r     = 0.15
    Cl_alpha = 0.0

    Cl = ( Cl_0
           + Cl_beta*beta_deg
           + Cl_alpha*alpha_deg
           + Cl_p*pb_2V
           + Cl_r*rb_2V
         )

    Cm_0     = -0.02
    Cm_alpha = -0.04
    Cm_q     = -8.0
    Cm = ( Cm_0
           + Cm_alpha*alpha_deg
           + Cm_q*qc_2V
         )

    Cn_0     = 0.0
    Cn_beta  = -0.05
    Cn_r     = -0.3
    Cn_p     = 0.07

    Cn = ( Cn_0
           + Cn_beta*beta_deg
           + Cn_r*rb_2V
           + Cn_p*pb_2V
         )

    # 5) Converte p/ força/momento no corpo
    # Eixo x_b = CA => + p/ frente
    X_b = CA * q_bar * S
    # Eixo y_b = + p/ direita
    Y_b = CY * q_bar * S
    # Eixo z_b (para baixo): CN>0 => força p/ cima => Z_b = -CN*qS
    Z_b = -CN * q_bar * S
    Faero_b = np.array([X_b, Y_b, Z_b])

    # Momentos
    L_b = Cl * q_bar * S * b
    M_b = Cm * q_bar * S * c
    N_b = Cn * q_bar * S * b
    Maero_b = np.array([L_b, M_b, N_b])

    # 6) Variáveis extras
    Yaero = np.array([CN, CA, CY, Cl, Cm, Cn, pb_2V, qc_2V, rb_2V])

    return Faero_b, Maero_b, Yaero
