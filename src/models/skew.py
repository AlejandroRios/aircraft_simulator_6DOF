"""
skew - Geração da Matriz Anti-Simétrica
Descrição: Gera a matriz anti-simétrica associada a um vetor 3D.
Entrada:
- n (numpy.ndarray): Vetor 3D.
Saída:
- n_tilde (numpy.ndarray): Matriz anti-simétrica 3x3.
"""

import numpy as np

def skew(n):
    """
    Gera a matriz anti-simétrica para o vetor dado.

    Parâmetros:
    - n: numpy.ndarray, vetor 3D.

    Retorno:
    - n_tilde: numpy.ndarray, matriz anti-simétrica 3x3.
    """
    if len(n) != 3:
        raise ValueError("O vetor deve ter exatamente 3 elementos.")

    n_tilde = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ])

    return n_tilde
