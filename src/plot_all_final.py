"""
plot_all_final - Geração de Gráficos de Resultados
Descrição: Plota gráficos de resultados da simulação, incluindo estados, controles e trajetórias.
Entrada:
- Tsol (numpy.ndarray): Vetor de tempos da simulação.
- Ysol (numpy.ndarray): Matriz de variáveis de saída (soluções).
- Usol (numpy.ndarray): Matriz de controles aplicados.
- Xsol (numpy.ndarray): Matriz de estados ao longo do tempo.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_all_final(Tsol, Ysol, Usol, Xsol):
    """
    Gera gráficos para visualizar os resultados da simulação.

    Parâmetros:
    - Tsol: numpy.ndarray, vetor de tempos da simulação.
    - Ysol: numpy.ndarray, matriz de variáveis de saída.
    - Usol: numpy.ndarray, matriz de controles aplicados.
    - Xsol: numpy.ndarray, matriz de estados ao longo do tempo.
    """
    print(f"Aviso: Ysol tem {Ysol.shape[1]} colunas disponíveis.")

    # Figura 1: Estados principais
    fig1, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].plot(Tsol, Ysol[:, 0])  # Substituímos por índices válidos
    axs[0, 0].set_xlabel('t [s]')
    axs[0, 0].set_ylabel('V [m/s]')

    axs[0, 1].plot(Tsol, Ysol[:, 1])
    axs[0, 1].set_xlabel('t [s]')
    axs[0, 1].set_ylabel('α [deg]')

    axs[0, 2].plot(Tsol, Ysol[:, 2])
    axs[0, 2].set_xlabel('t [s]')
    axs[0, 2].set_ylabel('q [deg/s]')

    axs[1, 0].plot(Tsol, Ysol[:, 3])
    axs[1, 0].set_xlabel('t [s]')
    axs[1, 0].set_ylabel('θ [deg]')

    axs[1, 1].plot(Tsol, Ysol[:, 4])
    axs[1, 1].set_xlabel('t [s]')
    axs[1, 1].set_ylabel('x [m]')

    axs[1, 2].plot(Tsol, Ysol[:, 5])
    axs[1, 2].set_xlabel('t [s]')
    axs[1, 2].set_ylabel('H [m]')

    fig1.tight_layout()
    fig1.savefig("figure_1.pdf")

    # Figura 2: Ângulos e velocidades
    fig2, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].plot(Tsol, Ysol[:, 6])
    axs[0, 0].set_xlabel('t [s]')
    axs[0, 0].set_ylabel('β [deg]')

    axs[0, 1].plot(Tsol, Ysol[:, 7])
    axs[0, 1].set_xlabel('t [s]')
    axs[0, 1].set_ylabel('φ [deg]')

    axs[0, 2].plot(Tsol, Ysol[:, 8])
    axs[0, 2].set_xlabel('t [s]')
    axs[0, 2].set_ylabel('p [deg/s]')

    axs[1, 0].plot(Tsol, Ysol[:, 9])
    axs[1, 0].set_xlabel('t [s]')
    axs[1, 0].set_ylabel('r [deg/s]')

    axs[1, 1].plot(Tsol, Ysol[:, 10])
    axs[1, 1].set_xlabel('t [s]')
    axs[1, 1].set_ylabel('ψ [deg]')

    axs[1, 2].plot(Tsol, Ysol[:, 11])
    axs[1, 2].set_xlabel('t [s]')
    axs[1, 2].set_ylabel('y [m]')

    fig2.tight_layout()
    fig2.savefig("figure_2.pdf")

    # Figura 3: Trajetória 3D
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot(Xsol[:, 11], Xsol[:, 5], Xsol[:, 4])  # Ajuste os índices de Xsol se necessário
    ax3.set_xlabel('y [m]')
    ax3.set_ylabel('x [m]')
    ax3.set_zlabel('H [m]')
    ax3.grid(True)
    ax3.set_title('Trajetória 3D')
    fig3.savefig("figure_3.pdf")

    # Figura 4: Controles
    fig4, axs = plt.subplots(2, 3, figsize=(12, 8))
    labels = ['δ_tle [N]', 'δ_tre [N]', 'δ_ih [deg]', 'δ_e [deg]', 'δ_a [deg]', 'δ_r [deg]']
    for i, ax in enumerate(axs.flatten()):
        ax.plot(Tsol, Usol[:, i])
        ax.set_xlabel('t [s]')
        ax.set_ylabel(labels[i])

    fig4.tight_layout()
    fig4.savefig("figure_4.pdf")

    plt.show()
