"""
Cmat - Geração de Matrizes de Rotação
Descrição: Gera a matriz de rotação para um eixo específico (1, 2, ou 3) ou um vetor arbitrário.
Entrada:
- n (int ou numpy.ndarray): Índice do eixo (1, 2, 3) ou vetor unitário (3 elementos).
- angle_rad (float): Ângulo de rotação em radianos.
Saída:
- C (numpy.ndarray): Matriz de rotação 3x3.
"""

import numpy as np
from models.skew import skew

def Cmat(n, angle_rad):
    """
    Gera a matriz de rotação.

    Parâmetros:
    - n: int ou numpy.ndarray, índice do eixo (1, 2, 3) ou vetor unitário.
    - angle_rad: float, ângulo de rotação em radianos.

    Retorno:
    - C: numpy.ndarray, matriz de rotação 3x3.
    """
    if isinstance(n, int):
        # Matrizes de rotação padrão para os eixos principais
        if n == 1:  # Rotação ao redor do eixo x
            C = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), np.sin(angle_rad)],
                [0, -np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif n == 2:  # Rotação ao redor do eixo y
            C = np.array([
                [np.cos(angle_rad), 0, -np.sin(angle_rad)],
                [0, 1, 0],
                [np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif n == 3:  # Rotação ao redor do eixo z
            C = np.array([
                [np.cos(angle_rad), np.sin(angle_rad), 0],
                [-np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("O índice do eixo deve ser 1, 2 ou 3.")
    else:
        # Rotação em torno de um vetor arbitrário
        n = n / np.linalg.norm(n)  # Normaliza o vetor
        C = (
            (1 - np.cos(angle_rad)) * np.outer(n, n) +
            np.cos(angle_rad) * np.eye(3) -
            np.sin(angle_rad) * skew(n)
        )

    return C
