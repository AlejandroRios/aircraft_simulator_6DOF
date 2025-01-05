import numpy as np

# -----------------------------------------------------------------------------
# globals.py
# -----------------------------------------------------------------------------

# Constante gravitacional
g = 9.80665  # m/s²

# Exemplo de dados para uma Mk82 (ajuste conforme referências oficiais)
# -----------------------------------------------------------------------------
m_bomba = 227.0  # Massa (kg) ~ 500 lb

# Momentos de inércia aproximados (kg·m²)
Ixx = 3.0     # Em torno do eixo longitudinal
Iyy = 75.0    # Eixo transversal
Izz = 75.0
Ixy = 0.0
Ixz = 0.0
Iyz = 0.0

J_O_bomba = np.array([
    [Ixx, -Ixy, -Ixz],
    [-Ixy, Iyy, -Iyz],
    [-Ixz, -Iyz, Izz]
])

# Seção transversal aproximada (por exemplo, diâmetro ~0,273 m)
# A bomba original do avião podia usar b=32.757, S=116, etc. 
# mas agora ajustamos a algo realista para a Mk82 ou outra "dumb bomb".

b_bomba = 0.273   # “Envergadura” ~ diâmetro (não é literal, mas mantido para compatibilidade)
S_bomba = np.pi*(0.273/2)**2  # Área frontal ~ 0,0585 m²
c_bomba = 0.273   # “corda” ~ diâmetro, para compatibilidade

# Vetor do CG (0,0,0) no corpo
rC_b = np.array([0.0, 0.0, 0.0])

# Posição “piloto” irrelevante, mas mantida para não quebrar o código
r_pilot_b = np.array([0.0, 0.0, 0.0])

# hex não se aplica (não há motor), mas podemos manter em 0
hex_bomba = 0.0

# Montando o dicionário "aircraft", mas agora representando a bomba.
aircraft = {
    'm': m_bomba,
    'J_O_b': J_O_bomba,
    'rC_b': rC_b,
    'b': b_bomba,
    'S': S_bomba,
    'c': c_bomba,
    'hex': hex_bomba,
    'r_pilot_b': r_pilot_b
}