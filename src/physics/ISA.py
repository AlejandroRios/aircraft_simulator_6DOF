"""
ISA - International Standard Atmosphere
Descrição: Calcula a atmosfera padrão internacional (ISA) em unidades SI.
Entrada:
- h (float): altitude (m)
Saídas:
- rho (float): densidade (kg/m³)
- T (float): temperatura (K)
- p (float): pressão (N/m²)
- a (float): velocidade do som (m/s)
"""

import numpy as np

def ISA(h):
    """
    Calcula os parâmetros atmosféricos padrão (ISA) para uma dada altitude.

    Parâmetros:
    - h: float, altitude em metros (m).

    Retorno:
    - rho: densidade do ar (kg/m³).
    - T: temperatura (K).
    - p: pressão atmosférica (N/m²).
    - a: velocidade do som (m/s).
    """
    # Conversão de metros para quilômetros
    h_km = h / 1000.0

    # Limites de altitude em quilômetros
    h1 = 11  # Troposfera
    h2 = 20  # Tropopausa
    h3 = 32  # Estratosfera

    # Gradientes de temperatura e constantes
    L0 = -6.5e-3  # Gradiente na troposfera (K/m)
    L2 = 1e-3     # Gradiente na estratosfera (K/m)
    g0 = 9.80665  # Aceleração gravitacional (m/s²)
    m0 = 28.96442  # Massa molar do ar (kg/kmol)
    R0 = 8314.32  # Constante universal dos gases (J/(kmol*K))
    R = R0 / m0   # Constante específica do ar (J/(kg*K))

    # Condições ao nível do mar
    T0 = 288.15  # Temperatura (K)
    p0 = 1.01325e5  # Pressão (N/m²)
    rho0 = 1.2250  # Densidade (kg/m³)

    # Condições na tropopausa (h = 11 km)
    T1 = T0 + L0 * h1 * 1000
    p1 = p0 * (T1 / T0) ** (-g0 / (R * L0))
    rho1 = rho0 * (T1 / T0) ** (-(1 + g0 / (R * L0)))

    # Condições na baixa estratosfera (h = 20 km)
    T2 = T1
    p2 = p1 * np.exp(-g0 / (R * T2) * (h2 - h1) * 1000)
    rho2 = rho1 * np.exp(-g0 / (R * T2) * (h2 - h1) * 1000)

    # Determinar os parâmetros atmosféricos em função da altitude
    if h_km <= h1:
        # Troposfera
        T = T0 + L0 * h * 1000
        p = p0 * (T / T0) ** (-g0 / (R * L0))
        rho = rho0 * (T / T0) ** (-(1 + g0 / (R * L0)))
    elif h_km <= h2:
        # Tropopausa
        T = T1
        p = p1 * np.exp(-g0 / (R * T) * (h_km - h1) * 1000)
        rho = rho1 * np.exp(-g0 / (R * T) * (h_km - h1) * 1000)
    elif h_km <= h3:
        # Estratosfera
        T = T2 + L2 * (h_km - h2) * 1000
        p = p2 * (T / T2) ** (-g0 / (R * L2))
        rho = rho2 * (T / T2) ** (-(1 + g0 / (R * L2)))
    else:
        raise ValueError("Altitude fora do intervalo suportado (0 a 32 km)")

    # Velocidade do som
    gamma = 1.4  # Razão de calores específicos (ar seco)
    a = np.sqrt(gamma * R * T)

    return rho, T, p, a
