"""
trimGNBA - Função de Trimagem
Descrição: Calcula as condições de equilíbrio (trim) da aeronave com base nos parâmetros fornecidos.
Entrada:
- x (numpy.ndarray): Vetor de incógnitas (14 elementos).
- trim_par (dict): Parâmetros de trimagem, incluindo velocidade, ângulos e forças.
Saída:
- f (numpy.ndarray): Resíduo das equações de equilíbrio.
"""

import numpy as np
from models.state_vec import state_vec
from models.control_vec import control_vec
from physics.dynamics import dynamics
from models.Cmat import Cmat

def trimGNBA(x, trim_par):
    # Certifique-se de que `x` é um vetor numpy
    x = np.atleast_1d(x)
    
    X = state_vec(x, trim_par)
    U = control_vec(x)

    Xdot, Y = dynamics(0, X, U, trim_par['W'])

    # Velocidade inercial
    C_tv = Cmat(2, np.radians(trim_par['gamma_deg'])) @ Cmat(3, np.radians(trim_par['chi_deg']))
    V_i = C_tv.T @ np.array([trim_par['V'], 0, 0])
    Beta = X[6]

    # Função objetivo
    f = np.array([
        Xdot[0],
        Xdot[1],
        Xdot[2],
        Xdot[3] - trim_par['theta_dot_deg_s'],
        Xdot[4] - V_i[2],
        Xdot[5] - V_i[0],
        Xdot[6],
        Xdot[7] - trim_par['phi_dot_deg_s'],
        Xdot[8],
        Xdot[9],
        Xdot[10] - trim_par['psi_dot_deg_s'],
        Xdot[11] - V_i[1],
        Beta,
        U[0] - U[1],
        0  
    ])
    return f