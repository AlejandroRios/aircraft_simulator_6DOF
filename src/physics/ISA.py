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

    # Faixas de altitude em km
    h1 = 11  # limite da Troposfera
    h2 = 20  # limite da Tropopausa
    h3 = 32  # limite da Estratosfera baixa

    # Gradientes e constantes
    L0 = -6.5e-3  # Gradiente na troposfera, em K/m
    L2 = 1.0e-3   # Gradiente na estratosfera (K/m)
    g0 = 9.80665  # Aceleração gravitacional (m/s²)
    m0 = 28.96442 # Massa molar do ar (kg/kmol)
    R0 = 8314.32  # Constante universal dos gases (J/(kmol*K))
    R  = R0 / m0  # Constante específica do ar (J/(kg*K))

    # Condições ao nível do mar
    T0   = 288.15    # Temperatura (K)
    p0   = 1.01325e5 # Pressão (N/m²)
    rho0 = 1.2250    # Densidade (kg/m³)

    # -------------------------------------------------------------
    # 1) Parâmetros na tropopausa (h = 11 km) -- calculados p/ uso posterior
    # -------------------------------------------------------------
    # Nesse ponto, h1*1000 = 11000 m
    T1 = T0 + L0 * (h1*1000)  # A ~ 216.65 K
    p1 = p0 * (T1 / T0) ** (-g0 / (R * L0))
    rho1 = rho0 * (T1 / T0) ** (-(1 + g0 / (R * L0)))

    # -------------------------------------------------------------
    # 2) Parâmetros na altitude de 20 km (início da estratosfera)
    # -------------------------------------------------------------
    # T2 é igual a T1, pois na tropopausa a T fica constante
    T2 = T1  
    # Diferencial em metros (de 11 km a 20 km = 9000 m)
    deltaH_trop = (h2 - h1) * 1000
    p2 = p1 * np.exp(-g0 / (R * T2) * deltaH_trop)
    rho2 = rho1 * np.exp(-g0 / (R * T2) * deltaH_trop)

    # -------------------------------------------------------------
    # 3) Determinar parâmetros de acordo com a altitude real
    # -------------------------------------------------------------
    if h_km <= h1:
        # ---------------- Troposfera (0 a 11 km) -----------------
        # T decresce linearmente com L0 = -6.5e-3 K/m
        T = T0 + L0 * h  # sem *1000, pois 'h' já está em metros
        p = p0 * (T / T0) ** (-g0 / (R * L0))
        rho = rho0 * (T / T0) ** (-(1 + g0 / (R * L0)))

    elif h_km <= h2:
        # ---------------- Tropopausa (11 a 20 km) ----------------
        # T é constante (T = T1), mas p e rho decaem exponencialmente
        deltaH = (h - h1 * 1000)  # diferença em metros acima de 11 km
        T = T1
        p = p1 * np.exp(-g0 / (R * T) * deltaH)
        rho = rho1 * np.exp(-g0 / (R * T) * deltaH)

    elif h_km <= h3:
        # ---------------- Estratosfera baixa (20 a 32 km) ----------------
        deltaH = (h_km - h2) * 1000  # diferença em metros acima de 20 km
        T = T2 + L2 * deltaH  # agora L2 = +1e-3 K/m
        p = p2 * (T / T2) ** (-g0 / (R * L2))
        rho = rho2 * (T / T2) ** (-(1 + g0 / (R * L2)))
    else:
        raise ValueError("Altitude fora do intervalo suportado (0 a 32 km)")

    # -------------------------------------------------------------
    # Velocidade do som
    # -------------------------------------------------------------
    gamma = 1.4  # Ar seco
    a = np.sqrt(gamma * R * T)

    return rho, T, p, a
