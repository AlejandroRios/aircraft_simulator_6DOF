import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Classe da rede para múltiplas saídas
class GeneralNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(GeneralNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Classe da rede para uma única saída
class SingleOutputNN(nn.Module):
    def __init__(self, input_size):
        super(SingleOutputNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Apenas uma saída
        )

    def forward(self, x):
        return self.layers(x)

# Função para treinar uma rede
def train_model(model, dataloaders, criterion, optimizer, epochs=1000):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Avaliação no conjunto de validação
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloaders['val']:
            outputs = model(X_batch)
            y_true.append(y_batch.numpy())
            y_pred.append(outputs.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return y_true, y_pred

# Função principal para comparar os modelos
def compare_models(data, input_columns, output_columns, epochs=1000):
    results = []

    # Preparar dados
    X = data[input_columns].values
    y = data[output_columns].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    input_size = len(input_columns)
    output_size = len(output_columns)

    # 1. Treinar a Rede Geral
    print("Training General Network...")
    general_model = GeneralNN(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(general_model.parameters(), lr=0.001)
    y_true_general, y_pred_general = train_model(general_model, dataloaders, criterion, optimizer, epochs)

    # Avaliar o desempenho da rede geral
    y_true_general = scaler_y.inverse_transform(y_true_general)
    y_pred_general = scaler_y.inverse_transform(y_pred_general)
    r2_general = r2_score(y_true_general, y_pred_general, multioutput='raw_values')

    print("General Network R2 Scores:", r2_general)

    # Redes para Saídas Individuais
    single_output_r2 = []
    for idx, output in enumerate(output_columns):
        print(f"Training Single-Output Network for {output}...")

        # Criar um scaler para a saída específica
        scaler_y_single = StandardScaler()
        y_train_single = scaler_y_single.fit_transform(y_train[:, idx].reshape(-1, 1))
        y_val_single = scaler_y_single.transform(y_val[:, idx].reshape(-1, 1))

        train_dataset_single = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train_single, dtype=torch.float32))
        val_dataset_single = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                            torch.tensor(y_val_single, dtype=torch.float32))

        dataloaders_single = {
            'train': DataLoader(train_dataset_single, batch_size=32, shuffle=True),
            'val': DataLoader(val_dataset_single, batch_size=32, shuffle=False)
        }

        # Treinar a rede específica
        single_model = SingleOutputNN(input_size)

        # Ajustar otimizador e taxa de aprendizado
        optimizer_single = torch.optim.Adam(single_model.parameters(), lr=0.0005)
        y_true_single, y_pred_single = train_model(single_model, dataloaders_single, criterion, optimizer_single, epochs)

        # Reverter a normalização
        y_true_single = scaler_y_single.inverse_transform(y_true_single)
        y_pred_single = scaler_y_single.inverse_transform(y_pred_single)

        # Calcular o R² para a saída específica
        r2_single = r2_score(y_true_single, y_pred_single)
        single_output_r2.append(r2_single)
        print(f"{output} R2 Score: {r2_single}")

    # Comparar Resultados
    results_df = pd.DataFrame({
        'Output': output_columns,
        'General_R2': r2_general,
        'Single_R2': single_output_r2
    })

    print(results_df)
    return results_df

# Carregar os dados
data = pd.read_csv('full_database.csv')
input_columns = ["ALPHA", "BETA", "MACH", "ALTITUDE", "DF1", "DF2", "DF3", "DF4"]
output_columns = ["CN", "CM", "CA", "CY", "CL", "CD", "CNA", "CMA", "CYB", "CLNB", "CLLB"]

# Comparar os modelos
results = compare_models(data, input_columns, output_columns, epochs=1000)

# Plotar os resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
width = 0.35  # Largura das barras
x = np.arange(len(output_columns))

plt.bar(x - width/2, results['General_R2'], width, label='General Network')
plt.bar(x + width/2, results['Single_R2'], width, label='Single Output Networks')

plt.xlabel('Outputs')
plt.ylabel('R2 Score')
plt.title('Comparison of R2 Scores')
plt.xticks(x, output_columns, rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
