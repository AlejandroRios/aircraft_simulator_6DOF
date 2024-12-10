import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from matplotlib.lines import Line2D

def plot_surface_comparison_superposed_distinct(model, scaler_X, scaler_y, data, fixed_conditions, x_var, y_var, output_var, input_columns, output_columns):
    # Extrair valores únicos da base para garantir mesmo shape
    unique_x = np.sort(data[x_var].unique())
    unique_y = np.sort(data[y_var].unique())
    X, Y = np.meshgrid(unique_x, unique_y)

    # Criar condições de entrada para cada ponto do grid
    input_conditions = []
    for i in range(len(X.flatten())):
        condition = fixed_conditions.copy()
        condition[x_var] = X.flatten()[i]
        condition[y_var] = Y.flatten()[i]
        input_conditions.append([condition[var] for var in input_columns])

    # Normalizar e prever
    input_scaled = scaler_X.transform(input_conditions)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(input_tensor).numpy()
    predictions = scaler_y.inverse_transform(predictions)

    # Índice da variável de saída
    out_idx = output_columns.index(output_var)
    Z_pred = predictions[:, out_idx].reshape(X.shape)

    # Criar tabela pivô para valores reais
    pivot = data.pivot_table(index=y_var, columns=x_var, values=output_var)
    Z_real = pivot.values

    # Ajustar o grid para corresponder às dimensões de Z_real
    X_grid, Y_grid = np.meshgrid(pivot.columns, pivot.index)

    # Plotar superfícies sobrepostas
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Superfície real (dados)
    surf_real = ax.plot_surface(X_grid, Y_grid, Z_real, cmap='Blues', alpha=0.9, edgecolor='k', linewidth=0.2)

    # Superfície predita (modelo)
    surf_pred = ax.plot_surface(X, Y, Z_pred, cmap='Reds', alpha=0.6, edgecolor='k', linewidth=0.2)

    # Criação de uma legenda manual
    line_real = Line2D([0], [0], marker='s', color='w', label='Real Data', 
                       markerfacecolor='blue', markersize=10)
    line_pred = Line2D([0], [0], marker='s', color='w', label='Prediction', 
                       markerfacecolor='red', markersize=10)
    ax.legend(handles=[line_real, line_pred], loc='best')

    ax.set_title(f"Comparação {output_var}: Dados Reais vs Predição")
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(output_var)

    plt.tight_layout()
    plt.show()

# Carregar modelo e scalers
model = torch.jit.load("models/CLLB_model_scripted.pt")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("models/CLLB_scaler_y.pkl")

# Carregar base de dados
data = pd.read_csv("full_database_10000.csv")

# Configurações
fixed_conditions = {"ALPHA": 0, "BETA": 0, "MACH": 0.85, "ALTITUDE": 5000, "DF1": 0, "DF2": 0, "DF3": 0, "DF4": 0}
input_columns = ["ALPHA", "BETA", "MACH", "ALTITUDE", "DF1", "DF2", "DF3", "DF4"]
output_columns = ["CN", "CM", "CA", "CY", "CL", "CD", "CNA", "CMA", "CYB", "CLNB", "CLLB"]
x_var = "BETA"
y_var = "MACH"
output_var = "CLLB"

plot_surface_comparison_superposed_distinct(model, scaler_X, scaler_y, data, fixed_conditions, x_var, y_var, output_var, input_columns, output_columns)
