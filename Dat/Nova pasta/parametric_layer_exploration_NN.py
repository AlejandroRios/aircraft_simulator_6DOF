import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Rede Neural parametrizável
class ParametricNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(ParametricNN, self).__init__()
        layers = []
        for neurons in hidden_layers:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(nn.ReLU())
            input_size = neurons
        layers.append(nn.Linear(input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Função de Treinamento e Avaliação
def train_and_evaluate(hidden_layers, dataloaders, input_size, output_size, scaler_y, epochs=1000):
    model = ParametricNN(input_size, output_size, hidden_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Treinamento
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Avaliação
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloaders['val']:
            outputs = model(X_batch)
            y_true.append(y_batch.numpy())
            y_pred.append(outputs.numpy())

    y_true = scaler_y.inverse_transform(np.vstack(y_true))
    y_pred = scaler_y.inverse_transform(np.vstack(y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return model, r2, mae

# Função para testar múltiplas combinações de neurônios
def validate_hidden_layers(data, input_columns, output_columns, layer_configurations, epochs=1000):
    results = []

    # Processar dados
    X = data[input_columns].values
    y = data[output_columns].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Dividir dados
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

    # Testar cada configuração
    for hidden_layers in layer_configurations:
        print(f"Testing configuration: {hidden_layers}")
        model, r2, mae = train_and_evaluate(hidden_layers, dataloaders, input_size, output_size, scaler_y, epochs=epochs)
        results.append({'hidden_layers': hidden_layers, 'R2': r2, 'MAE': mae})

    return pd.DataFrame(results)

# Configurações de teste
data = pd.read_csv('full_database.csv')
input_columns = ["ALPHA", "BETA", "MACH", "ALTITUDE", "DF1", "DF2", "DF3", "DF4"]
output_columns = ["CN", "CM", "CA", "CY", "CL", "CD", "CNA", "CMA", "CYB", "CLNB", "CLLB"]

layer_configurations = [
    [32],
    [64],
    [128],
    [256],
    [64, 32],
    [128, 64],
    [256, 128],
    [128, 128],
    [128, 64, 32]
]

# Validar as configurações
results = validate_hidden_layers(data, input_columns, output_columns, layer_configurations, epochs=300)

# Exibir os resultados
print(results)

# Plotar os resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for index, row in results.iterrows():
    plt.scatter(len(row['hidden_layers']), row['R2'], label=f"{row['hidden_layers']}", s=100)
plt.xlabel('Number of Layers')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Hidden Layer Configuration')
plt.legend()
plt.grid()
plt.show()
