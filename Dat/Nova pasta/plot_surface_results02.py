import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from matplotlib.lines import Line2D

def plot_surface_comparison(model, scaler_X, scaler_y, data, fixed_conditions, x_var, y_var, output_var, input_columns):
    """
    Plota a superfície predita pelo modelo e compara com os dados reais.
    
    Args:
        model: Modelo carregado para a saída específica.
        scaler_X: Scaler para normalização das entradas.
        scaler_y: Scaler para normalização da saída.
        data: DataFrame com os dados originais.
        fixed_conditions: Dicionário com as variáveis fixas e seus valores.
        x_var: Nome da variável no eixo X.
        y_var: Nome da variável no eixo Y.
        output_var: Nome da saída desejada.
        input_columns: Lista das colunas de entrada usadas no treinamento.
    """
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

    # Reformatar a predição para o grid
    Z_pred = predictions.reshape(X.shape)

    # Criar tabela pivô para valores reais
    pivot = data.pivot_table(index=y_var, columns=x_var, values=output_var)
    Z_real = pivot.values

    # Ajustar o grid para corresponder às dimensões de Z_real
    X_grid, Y_grid = np.meshgrid(pivot.columns, pivot.index)

    # Plotar superfícies sobrepostas
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Superfície real (dados)
    surf_real = ax.plot_surface(X_grid, Y_grid, Z_real, cmap='Blues', alpha=0.2, edgecolor='k', linewidth=0.2)

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


def main(output_var, x_var, y_var, fixed_conditions, data_path="full_database_10000.csv", model_dir="models"):
    """
    Função principal para carregar modelo, scaler e plotar a comparação.
    
    Args:
        output_var: Nome da saída (e.g., 'CLLB').
        x_var: Nome da variável no eixo X.
        y_var: Nome da variável no eixo Y.
        fixed_conditions: Dicionário com as condições fixas.
        data_path: Caminho para o arquivo CSV com os dados originais.
        model_dir: Diretório onde os modelos e scalers estão salvos.
    """
    # Carregar os dados
    data = pd.read_csv(data_path)

    # Colunas de entrada
    input_columns = ["ALPHA", "BETA", "MACH", "ALTITUDE", "DF1", "DF2", "DF3", "DF4"]

    # Carregar scaler de entrada
    scaler_X = joblib.load(f"scaler_X.pkl")

    # Carregar modelo e scaler para a saída desejada
    model_path = f"{model_dir}/{output_var}_model_scripted.pt"
    scaler_y_path = f"{model_dir}/{output_var}_scaler_y.pkl"

    model = torch.jit.load(model_path)
    scaler_y = joblib.load(scaler_y_path)

    print(f"Modelo e scaler carregados para a saída: {output_var}")

    # Plotar comparação
    plot_surface_comparison(
        model, scaler_X, scaler_y, data, fixed_conditions, x_var, y_var, output_var, input_columns
    )


# Exemplo de uso
if __name__ == "__main__":
    # Configurações
    output_var = "CLLB"  # Variável de saída a ser analisada
    x_var = "BETA"       # Variável no eixo X
    y_var = "MACH"       # Variável no eixo Y
    fixed_conditions = {"ALPHA": 0, "BETA": 0, "MACH": 0.85, "ALTITUDE": 5000, "DF1": 0, "DF2": 0, "DF3": 0, "DF4": 0}

    # Caminhos para os dados e modelos
    data_path = "full_database_10000.csv"
    model_dir = "models"

    # Executar
    main(output_var, x_var, y_var, fixed_conditions, data_path, model_dir)
