"""
state_vec - Função para gerar o vetor de estado da aeronave
Descrição: Constrói o vetor de estado completo a partir do vetor de incógnitas e parâmetros de trimagem.
Entrada:
- x (numpy.ndarray): Vetor de incógnitas (15 elementos).
- trim_par (dict): Parâmetros de trimagem, incluindo altitude e velocidade.
Saída:
- X (numpy.ndarray): Vetor de estado completo (12 elementos).
"""

import numpy as np

def state_vec(x, trim_par):
    """
    Gera o vetor de estado da aeronave.

    Parâmetros:
    - x: numpy.ndarray, vetor de incógnitas (15 elementos).
    - trim_par: dict, parâmetros de trimagem, incluindo altitude (H_m).

    Retorno:
    - X: numpy.ndarray, vetor de estado completo (12 elementos).
    """
    # Inicialização do vetor de estado
    X = np.zeros(12)

    print(X)

    # Estados diretamente extraídos de 'x'
    X[0] = x[0]  # Velocidade (V)
    X[1] = x[1]  # Ângulo de ataque (alpha)
    X[2] = x[2]  # Velocidade angular pitch (q)
    X[3] = x[3]  # Ângulo de inclinação (theta)

    # Estado configurado pelo parâmetro de trimagem
    X[4] = trim_par['H_m']  # Altitude (H)

    # Outros estados a partir de 'x'
    X[6] = x[4]  # Ângulo de deriva (beta)
    X[7] = x[5]  # Ângulo de rolamento (phi)
    X[8] = x[6]  # Velocidade angular roll (p)
    X[9] = x[7]  # Velocidade angular yaw (r)
    X[10] = x[8]  # Ângulo yaw (psi)

    return X
