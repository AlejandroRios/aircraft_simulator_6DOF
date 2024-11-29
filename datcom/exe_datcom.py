import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyDOE2 import lhs
import numpy as np

from input_generator import MissileDATCOMInput
from output_reader import DATCOMResultReader

# Gerar o DOE
def generate_doe_pydoe(n_samples, bounds):
    """
    Gera um DOE usando Latin Hypercube Sampling (LHS) com pyDOE2.
    
    Parâmetros:
    - n_samples: Número de amostras.
    - bounds: Lista de limites inferiores e superiores [(lb1, ub1), (lb2, ub2), ...].
    
    Retorna:
    - Um DataFrame com os valores de entrada gerados.
    """
    n_inputs = len(bounds)
    X = lhs(n_inputs, samples=n_samples)  # Latin Hypercube Sampling

    # Escalando para os limites reais
    scaled_X = np.zeros_like(X)
    for i, (lb, ub) in enumerate(bounds):
        scaled_X[:, i] = lb + (ub - lb) * X[:, i]
    
    # Criar DataFrame com os parâmetros
    columns = ["MACH", "ALTITUDE", "ALPHA"]
    doe_df = pd.DataFrame(scaled_X, columns=columns)
    return doe_df


# Função para rodar o DATCOM para cada condição de voo
def run_datcom_for_conditions(conditions_df, datcom_input_path="for005.dat", datcom_output_path="for006.dat"):
    results = []

    for _, row in conditions_df.iterrows():
        # Gerar o arquivo de entrada for005.dat
        datcom_input = MissileDATCOMInput(
            mach_vals=[row['MACH']],
            alt_vals=[row['ALTITUDE']],
            alpha_vals=[row['ALPHA']]
        )
        datcom_input.gerar_input(datcom_input_path)

        # Executar o DATCOM (substitua pelo comando correto no seu sistema)
        os.system(f"missile_datcom.exe < {datcom_input_path}")

        # Ler os resultados do for006.dat
        reader = DATCOMResultReader(datcom_output_path)
        reader.read_file()

        # Coletar os coeficientes do primeiro caso
        dataframes = reader.get_dataframes()
        for dtype, df in dataframes:
            if dtype == "LONGITUDINAL":
                row_results = {
                    "MACH": row["MACH"],
                    "ALTITUDE": row["ALTITUDE"],
                    "ALPHA": row["ALPHA"],
                    "CL": float(df["CL"].iloc[0]),
                    "CD": float(df["CD"].iloc[0]),
                }
            elif dtype == 'DERIVATIVES':
                row_results = {
                    "CM": float(df["CM"].iloc[0]),
                    "CYB": float(df["CYB"].iloc[0])
                }                
                results.append(row_results)

    # Retornar todos os resultados
    return pd.DataFrame(results)


# Função para treinamento de rede neural
def train_neural_network(data):
    # Preparar os dados
    X = data[["MACH", "ALTITUDE", "ALPHA"]].values
    y = data[["CL", "CD", "CM"]].values

    # Normalizar os dados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Dividir em treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir a rede neural
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)  # 3 saídas: CL, CD, CM
            )

        def forward(self, x):
            return self.layers(x)

    # Instanciar e treinar a rede
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Converter para tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Treinamento
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validação
        model.eval()
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    return model, scaler_X, scaler_y


# Exemplo de uso
# Definir os limites para MACH, ALTITUDE e ALPHA
bounds = [
    (0.5, 2.5),      # MACH: de 0.5 a 2.5
    (0, 20000),      # ALTITUDE: de 0 a 20.000 ft
    (-5, 20)         # ALPHA: de -5° a 20°
]

# Gerar o DOE com 100 amostras
doe_data = generate_doe_pydoe(n_samples=100, bounds=bounds)

# Execute o DATCOM para obter os resultados
results_df = run_datcom_for_conditions(doe_data)

# Treinamento da rede neural
model, scaler_X, scaler_y = train_neural_network(results_df)

# Agora, você pode usar o modelo para fazer previsões com novas condições de voo
def predict_coefficients(model, scaler_X, scaler_y, mach, altitude, alpha):
    input_data = [[mach, altitude, alpha]]
    input_scaled = scaler_X.transform(input_data)
    prediction = model(torch.tensor(input_scaled, dtype=torch.float32))
    return scaler_y.inverse_transform(prediction.detach().numpy())

# Exemplo de previsão para Mach=2.0, Altitude=10000, Alpha=5
predicted_coefficients = predict_coefficients(model, scaler_X, scaler_y, 2.0, 10000, 5)
print("Predicted Coefficients (CL, CD, CM):", predicted_coefficients)
