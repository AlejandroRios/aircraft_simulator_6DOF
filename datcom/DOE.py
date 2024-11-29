from pyDOE2 import lhs
import numpy as np
import pandas as pd

# Definir os limites das variáveis
mach_range = [0.5, 3.0]
altitude_range = [0, 20000]  # em pés
alpha_range = [-5, 20]  # em graus

# Número de amostras
n_samples = 100  # Ajuste conforme necessário

# Gerar amostras com Cubo Hipercúbico Latino
lhs_samples = lhs(3, samples=n_samples)  # 3 variáveis: Mach, Altitude, Alpha

# Escalar as amostras para os limites definidos
mach_samples = lhs_samples[:, 0] * (mach_range[1] - mach_range[0]) + mach_range[0]
altitude_samples = lhs_samples[:, 1] * (altitude_range[1] - altitude_range[0]) + altitude_range[0]
alpha_samples = lhs_samples[:, 2] * (alpha_range[1] - alpha_range[0]) + alpha_range[0]

# Criar DataFrame
doe_data = pd.DataFrame({
    'MACH': mach_samples,
    'ALTITUDE': altitude_samples,
    'ALPHA': alpha_samples
})

print(doe_data.head())
