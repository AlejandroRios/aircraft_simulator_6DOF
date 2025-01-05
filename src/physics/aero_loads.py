"""
aero_loads - Cálculo das Cargas Aerodinâmicas em eixo de corpo (body axis)
com coeficientes DATCOM, incluindo derivações p*b/(2V), q*c/(2V), etc.
EM GRAUS (alpha_deg, beta_deg)
"""

import numpy as np
from models.Cmat import Cmat  # se ainda precisar de alguma rotação matricial
from physics.ISA import ISA
from globals import aircraft, g

def aero_loads(X, U):
    """
    Calcula as forças (Faero_b) e momentos (Maero_O_b) aerodinâmicos no eixo de corpo,
    usando coeficientes DATCOM em função de ângulos em graus.

    Parâmetros:
    - X (np.ndarray): Vetor de estados (12 elementos) no formato:
        [ V, alpha_deg, q_deg_s, theta_deg, H_m, x, beta_deg, phi_deg,
          p_deg_s, r_deg_s, psi_deg, y ]
      Todos os ângulos (alpha, beta, etc.) e velocidades angulares (p, q, r) em deg/s.
    - U (np.ndarray): Vetor de controles. Para bomba “dumb”, normalmente zeros,
      mas deixado para compatibilidade.

    Retorno:
    - Faero_b (np.ndarray(3)): [X_b, Y_b, Z_b] em newtons, no referencial corpo.
    - Maero_O_b (np.ndarray(3)): [L_b, M_b, N_b] em newton-metro, no corpo (torno do CG).
    - Yaero (np.ndarray): Vetor com variáveis extras, ex. [C_N, C_A, C_Y, C_l, C_m, C_n].
    """

    # ---------------------------------------------------------------------
    # 1) Extrair estados de X, que estão em GRAUS (para alpha, beta, p, q, r)
    # ---------------------------------------------------------------------
    V        = X[0]
    alpha_deg= X[1]
    q_deg_s  = X[2]
    H_m      = X[4]
    beta_deg = X[6]
    p_deg_s  = X[8]
    r_deg_s  = X[9]

    # Convertemos APENAS para rad/s as velocidades angulares, pois p, q, r
    # em deg/s não servem para pb/(2V), etc. Precisamos de radianos/s no escalonamento.
    p_rad_s = np.deg2rad(p_deg_s)
    q_rad_s = np.deg2rad(q_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)

    # alpha e beta continuam em graus para as derivadas do DATCOM:
    # Se você precisar de seno/cosseno de alpha, então converterá para rad.
    alpha_rad = np.deg2rad(alpha_deg)  # p/ trigonometria
    beta_rad  = np.deg2rad(beta_deg)

    # ---------------------------------------------------------------------
    # 2) Parâmetros geométricos e aeronáuticos
    # ---------------------------------------------------------------------
    m  = aircraft['m']   # Massa se precisar
    S  = aircraft['S']   # Área de referência
    b  = aircraft['b']   # Comprimento de referência p/ rolagem (diâmetro)
    c  = aircraft['c']   # Comprimento de referência p/ arfagem (diâmetro)

    # ---------------------------------------------------------------------
    # 3) Pressão dinâmica e Mach
    # ---------------------------------------------------------------------
    rho, _, _, a = ISA(H_m)
    q_bar = 0.5 * rho * (V**2)
    Mach = V / a if a > 1e-6 else 0.0

    # Fatores adimensionais (clássicos do DATCOM) p, q, r normalizados:
    # Precisamos de p, q, r em rad/s, mas b, c, V estão em unidades SI
    pb_2V = (p_rad_s * b) / (2.0 * V) if V>1e-6 else 0.0
    qc_2V = (q_rad_s * c) / (2.0 * V) if V>1e-6 else 0.0
    rb_2V = (r_rad_s * b) / (2.0 * V) if V>1e-6 else 0.0

    # ---------------------------------------------------------------------
    # 4) Calcular coeficientes DATCOM no Eixo de Corpo (em GRAUS)
    # ---------------------------------------------------------------------
    # Exemplo genérico. Substitua pelos valores/tabelas do seu DATCOM:
    # Atenção: As derivadas C_Nalpha, C_Nbeta etc. assumem alpha_deg e beta_deg em GRAUS.

    # Força Normal
    CN_0       = 0.20    # CN em alpha=0, beta=0, Mach fixo, ...
    CN_alpha   = 0.03    # dCN/dalpha (1/deg)
    CN_q       = 3.2     # dCN/d(qc/2V)
    CN_beta    = 0.01    # dCN/dbeta (1/deg) se houver
    CN_p       = 0.1
    CN_r       = 0.05

    CN = ( CN_0
           + CN_alpha*(alpha_deg)
           + CN_q*(qc_2V)
           + CN_beta*(beta_deg)
           + CN_p*(pb_2V)
           + CN_r*(rb_2V)
         )

    # Força Axial
    CA_0       = 0.05
    CA_alpha   = 0.002   # (1/deg)
    CA_q       = 0.0
    CA_beta    = 0.0
    CA_p       = 0.0
    CA_r       = 0.0

    CA = ( CA_0
           + CA_alpha*(alpha_deg)
           + CA_q*(qc_2V)
           + CA_beta*(beta_deg)
           + CA_p*(pb_2V)
           + CA_r*(rb_2V)
         )

    # Força Lateral
    CY_0       = 0.0
    CY_beta    = 0.02    # (1/deg)
    CY_p       = -0.2
    CY_r       = 0.3
    CY_alpha   = 0.0
    CY_q       = 0.0

    CY = ( CY_0
           + CY_beta*(beta_deg)
           + CY_alpha*(alpha_deg)
           + CY_p*(pb_2V)
           + CY_q*(qc_2V)
           + CY_r*(rb_2V)
         )

    # Momento de Rolagem (C_l)
    Cl_0       = 0.0
    Cl_beta    = 0.00
    Cl_p       = -0.40
    Cl_r       = 0.15
    Cl_alpha   = 0.0
    Cl_q       = 0.0

    Cl = ( Cl_0
           + Cl_beta*(beta_deg)
           + Cl_alpha*(alpha_deg)
           + Cl_p*(pb_2V)
           + Cl_q*(qc_2V)
           + Cl_r*(rb_2V)
         )

    # Momento de arfagem (C_m)
    Cm_0       = -0.02
    Cm_alpha   = -0.04  # (1/deg)
    Cm_q       = -8.0   # dCm/d(qc/2V)
    Cm_beta    =  0.0
    Cm_p       =  0.0
    Cm_r       =  0.0

    Cm = ( Cm_0
           + Cm_alpha*(alpha_deg)
           + Cm_q*(qc_2V)
           + Cm_beta*(beta_deg)
           + Cm_p*(pb_2V)
           + Cm_r*(rb_2V)
         )

    # Momento de guinada (C_n)
    Cn_0       = 0.0
    Cn_beta    = -0.05  # (1/deg)
    Cn_r       = -0.3
    Cn_p       =  0.07
    Cn_alpha   =  0.00
    Cn_q       =  0.00

    Cn = ( Cn_0
           + Cn_beta*(beta_deg)
           + Cn_alpha*(alpha_deg)
           + Cn_p*(pb_2V)
           + Cn_q*(qc_2V)
           + Cn_r*(rb_2V)
         )

    # ---------------------------------------------------------------------
    # 5) Converter coeficientes para forças e momentos no corpo
    # ---------------------------------------------------------------------
    # Convenção DATCOM Body Axis:
    #   X_b =  CA * q_bar * S
    #   Y_b =  CY * q_bar * S
    #   Z_b = -CN * q_bar * S   (para "cima" se CN>0, mas eixos z_B pointing down)
    X_b = CA * q_bar * S
    Y_b = CY * q_bar * S
    Z_b = -CN * q_bar * S
    Faero_b = np.array([X_b, Y_b, Z_b])

    # Momentos:
    #   L_b = Cl * q_bar * S * b
    #   M_b = Cm * q_bar * S * c
    #   N_b = Cn * q_bar * S * b
    L_b = Cl * q_bar * S * b
    M_b = Cm * q_bar * S * c
    N_b = Cn * q_bar * S * b
    Maero_O_b = np.array([L_b, M_b, N_b])

    # ---------------------------------------------------------------------
    # 6) Variáveis extras (opcional)
    # ---------------------------------------------------------------------
    # Aqui podemos retornar [CN, CA, CY, Cl, Cm, Cn, pb_2V, qc_2V, rb_2V] para debugging
    Yaero = np.array([CN, CA, CY, Cl, Cm, Cn, pb_2V, qc_2V, rb_2V])

    return Faero_b, Maero_O_b, Yaero
