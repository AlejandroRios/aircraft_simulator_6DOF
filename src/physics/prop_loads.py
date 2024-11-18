"""
prop_loads - Cálculo das Cargas Propulsivas
Descrição: Calcula as forças e momentos propulsivos no referencial corpo e variáveis relacionadas.
Entrada:
- X (numpy.ndarray): Vetor de estados da aeronave.
- U (numpy.ndarray): Vetor de controles.
Saída:
- Fprop_b (numpy.ndarray): Forças propulsivas no referencial corpo.
- Mprop_O_b (numpy.ndarray): Momentos propulsivos no referencial corpo em relação ao CG.
- Yprop (numpy.ndarray): Variáveis adicionais relacionadas às cargas propulsivas.
"""

import numpy as np
from models.Cmat import Cmat
from physics.ISA import ISA
def prop_loads(X, U):
    """
    Calcula as cargas propulsivas.

    Parâmetros:
    - X: numpy.ndarray, vetor de estados da aeronave.
    - U: numpy.ndarray, vetor de controles.

    Retorno:
    - Fprop_b: numpy.ndarray, forças propulsivas no referencial corpo.
    - Mprop_O_b: numpy.ndarray, momentos propulsivos no referencial corpo.
    - Yprop: numpy.ndarray, variáveis adicionais (forças individuais, momentos individuais).
    """
    # Estados extraídos de X
    H_m = X[4]
    V = X[0]

    # Condições atmosféricas e cálculo de Mach
    _, _, _, a = ISA(H_m)
    Mach = V / a

    # Configuração do motor esquerdo
    ile_rad = np.deg2rad(2)  # Inclinação
    taule_rad = np.deg2rad(1.5)  # Torção
    rle_b = np.array([4.899, -5.064, 1.435])  # Posição no referencial corpo

    Fx_le = U[0]
    Fy_le = 0
    Fz_le = 0

    Mt_le = Cmat(2, ile_rad) @ Cmat(3, taule_rad)
    Fle = np.array([Fx_le, Fy_le, Fz_le])
    Fble = Mt_le @ Fle
    Mble = np.cross(rle_b, Fble)

    # Configuração do motor direito
    ire_rad = np.deg2rad(2)  # Inclinação
    taure_rad = np.deg2rad(-1.5)  # Torção
    rre_b = np.array([4.899, 5.064, 1.435])  # Posição no referencial corpo

    Fx_re = U[1]
    Fy_re = 0
    Fz_re = 0

    Mt_re = Cmat(2, ire_rad) @ Cmat(3, taure_rad)
    Fre = np.array([Fx_re, Fy_re, Fz_re])
    Fbre = Mt_re @ Fre
    Mbre = np.cross(rre_b, Fbre)

    # Soma de forças e momentos
    Fprop_b = Fble + Fbre  # Soma das forças
    Mprop_O_b = Mble + Mbre  # Soma dos momentos

    # Variáveis adicionais de saída
    Yprop = np.concatenate([Fble, Fbre, Mble, Mbre])

    return Fprop_b, Mprop_O_b, Yprop
