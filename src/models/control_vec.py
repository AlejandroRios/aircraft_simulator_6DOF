"""
control_vec - Função para gerar o vetor de controle da aeronave
Descrição: Constrói o vetor de controle a partir do vetor de incógnitas.
Entrada:
- x (numpy.ndarray): Vetor de incógnitas (15 elementos).
Saída:
- U (numpy.ndarray): Vetor de controle (6 elementos).
"""

import numpy as np

def control_vec(x):
    """
    Gera o vetor de controle da aeronave.

    Parâmetros:
    - x: numpy.ndarray, vetor de incógnitas (15 elementos).

    Retorno:
    - U: numpy.ndarray, vetor de controle (6 elementos).
    """
    # Inicialização do vetor de controle
    U = np.zeros(6)

    # Controles diretamente extraídos de 'x'
    U[0] = x[9]  # Empuxo da esquerda (Tle)
    U[1] = x[10]  # Empuxo da direita (Tre)
    U[2] = x[11]  # Ângulo do estabilizador horizontal (ih)
    U[3] = x[12]  # Deflexão do profundor (de)
    U[4] = x[13]  # Deflexão dos ailerons (da)
    U[5] = x[14]  # Deflexão do leme de direção (dr)

    return U
