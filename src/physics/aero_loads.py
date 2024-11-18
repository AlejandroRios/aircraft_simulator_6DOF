"""
aero_loads - Cálculo das Cargas Aerodinâmicas
Descrição: Calcula as forças e momentos aerodinâmicos no referencial corpo e outras variáveis relacionadas.
Entrada:
- X (numpy.ndarray): Vetor de estados da aeronave.
- U (numpy.ndarray): Vetor de controles.
Saída:
- Faero_b (numpy.ndarray): Forças aerodinâmicas no referencial corpo.
- Maero_O_b (numpy.ndarray): Momentos aerodinâmicos no referencial corpo em relação ao CG.
- Yaero (numpy.ndarray): Variáveis aerodinâmicas adicionais.
"""

import numpy as np
from models.Cmat import Cmat
from physics.ISA import ISA
from globals import aircraft, g

def aero_loads(X, U):
    """
    Calcula as cargas aerodinâmicas.

    Parâmetros:
    - X: numpy.ndarray, vetor de estados da aeronave.
    - U: numpy.ndarray, vetor de controles.

    Retorno:
    - Faero_b: numpy.ndarray, forças aerodinâmicas no referencial corpo.
    - Maero_O_b: numpy.ndarray, momentos aerodinâmicos no referencial corpo.
    - Yaero: numpy.ndarray, variáveis adicionais (sustentação, arrasto, etc.).
    """
    global aircraft

    # Estados extraídos de X
    V = X[0]
    alpha_deg = X[1]
    beta_deg = X[6]
    p_deg_s = X[8]
    q_deg_s = X[2]
    r_deg_s = X[9]
    H_m = X[4]

    # Controles extraídos de U
    ih_deg = U[2]
    de_deg = U[3]
    da_deg = U[4]
    dr_deg = U[5]

    # Conversões para radianos
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)
    p_rad_s = np.deg2rad(p_deg_s)
    q_rad_s = np.deg2rad(q_deg_s)
    r_rad_s = np.deg2rad(r_deg_s)

    # Propriedades da aeronave
    b = aircraft['b']
    c = aircraft['c']
    S = aircraft['S']

    # Coeficientes aerodinâmicos
    CL0, CLalpha, CLq, CLih, CLde = 0.308, 0.133, 16.7, 0.0194, 0.00895
    CD0, CDalpha, CDalpha2 = 0.02207, 0.00271, 0.000603
    CDbeta2, CDp2, CDr2 = 0.000160, 0.5167, 0.5738
    CYbeta, CYp, CYr, CYda, CYdr = 0.0228, 0.084, -1.21, 2.36e-4, -5.75e-3
    Cm0, Cmalpha, Cmq, Cmih, Cmde = 0.0170, -0.0402, -57.0, -0.0935, -0.0448
    Clbeta, Clp, Clr, Clda, Cldr = -3.66e-3, -0.661, 0.144, -2.87e-3, 6.75e-4
    Cnbeta, Cnp, Cnr, Cnda, Cndr = 5.06e-3, 0.0219, -0.634, 0, -3.26e-3

    # Coeficientes de sustentação, arrasto e força lateral
    CL = CL0 + CLalpha * alpha_deg + CLq * (q_rad_s * c / (2 * V)) + CLih * ih_deg + CLde * de_deg
    CD = CD0 + CDalpha * alpha_deg + CDalpha2 * alpha_deg**2
    CD += CDbeta2 * beta_deg**2 + CDp2 * (p_rad_s * b / (2 * V))**2 + CDr2 * (r_rad_s * b / (2 * V))**2
    CY = CYbeta * beta_deg + CYp * (p_rad_s * b / (2 * V)) + CYr * (r_rad_s * b / (2 * V))
    CY += CYda * da_deg + CYdr * dr_deg

    # Momentos aerodinâmicos
    Cm = Cm0 + Cmalpha * alpha_deg + Cmq * (q_rad_s * c / (2 * V)) + Cmih * ih_deg + Cmde * de_deg
    Cl = Clbeta * beta_deg + Clp * (p_rad_s * b / (2 * V)) + Clr * (r_rad_s * b / (2 * V))
    Cl += Clda * da_deg + Cldr * dr_deg
    Cn = Cnbeta * beta_deg + Cnp * (p_rad_s * b / (2 * V)) + Cnr * (r_rad_s * b / (2 * V))
    Cn += Cnda * da_deg + Cndr * dr_deg

    # Condições atmosféricas
    rho, _, _, _ = ISA(H_m)
    q_bar = 0.5 * rho * V**2

    # Forças aerodinâmicas
    L = q_bar * S * CL
    D = q_bar * S * CD
    Y = q_bar * S * CY

    # Momentos aerodinâmicos
    La = q_bar * S * b * Cl
    Ma = q_bar * S * c * Cm
    Na = q_bar * S * b * Cn

    # Transformação para o referencial corpo
    Faero_b = Cmat(2, alpha_rad) @ Cmat(3, -beta_rad) @ np.array([-D, -Y, -L])
    Maero_O_b = np.array([La, Ma, Na])

    # Variáveis adicionais
    Yaero = np.array([L, D, Y, La, Ma, Na])

    return Faero_b, Maero_O_b, Yaero
