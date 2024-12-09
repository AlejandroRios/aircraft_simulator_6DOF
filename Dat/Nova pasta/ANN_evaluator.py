import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# Função para treinar um modelo com Early Stopping
def train_model_with_early_stopping(model, dataloaders, criterion, optimizer, num_epochs=1000, patience=50):
    best_model = None
    best_loss = float('inf')
    no_improvement = 0

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Treinamento
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloaders['val']:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(dataloaders['val'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Restaurar o melhor modelo
    if best_model is not None:
        model.load_state_dict(best_model)

    # Plotar as perdas
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid()
    plt.show()

    return model


# Função principal para comparar os modelos
def compare_models(data, input_columns, output_columns, num_epochs=1000, patience=50):
    results = []

    # Preparar os dados
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
    general_model = train_model_with_early_stopping(general_model, dataloaders, criterion, optimizer, num_epochs, patience)

    # Avaliar o desempenho da rede geral
    general_model.eval()
    y_true_general, y_pred_general = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloaders['val']:
            outputs = general_model(X_batch)
            y_true_general.append(y_batch.numpy())
            y_pred_general.append(outputs.numpy())

    y_true_general = scaler_y.inverse_transform(np.vstack(y_true_general))
    y_pred_general = scaler_y.inverse_transform(np.vstack(y_pred_general))
    r2_general = r2_score(y_true_general, y_pred_general, multioutput='raw_values')

    print("General Network R2 Scores:", r2_general)

    # 2. Treinar Redes para Saídas Individuais
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
        optimizer_single = torch.optim.Adam(single_model.parameters(), lr=0.001)
        single_model = train_model_with_early_stopping(single_model, dataloaders_single, criterion, optimizer_single, num_epochs, patience)

        # Avaliar a rede específica
        single_model.eval()
        y_true_single, y_pred_single = [], []
        with torch.no_grad():
            for X_batch, y_batch in dataloaders_single['val']:
                outputs = single_model(X_batch)
                y_true_single.append(y_batch.numpy())
                y_pred_single.append(outputs.numpy())

        y_true_single = scaler_y_single.inverse_transform(np.vstack(y_true_single))
        y_pred_single = scaler_y_single.inverse_transform(np.vstack(y_pred_single))
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
data = pd.read_csv('full_database_5000.csv')
input_columns = ["ALPHA", "BETA", "MACH", "ALTITUDE", "DF1", "DF2", "DF3", "DF4"]
output_columns = ["CN", "CM", "CA", "CY", "CL", "CD", "CNA", "CMA", "CYB", "CLNB", "CLLB"]

# Comparar os modelos
results = compare_models(data, input_columns, output_columns, num_epochs=1000, patience=50)

# Plotar os resultados
plt.figure(figsize=(10, 6))
width = 0.35
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
