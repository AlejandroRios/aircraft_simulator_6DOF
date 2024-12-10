import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# Define a arquitetura da rede neural
class AerodynamicNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(AerodynamicNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.layers(x)


# Função para treinar o modelo
def train_model(model, dataloaders, criterion, optimizer, num_epochs=500, patience=50):
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
                print("Early stopping triggered.")
                break

    # Restaurar o melhor modelo
    model.load_state_dict(best_model)

    # Plota as perdas
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid()
    plt.show()

    return model


# Função para avaliar o modelo
def evaluate_model(model, dataloader, scaler_y, output_name):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            y_true.append(y_batch.numpy())
            y_pred.append(outputs.numpy())

    y_true = scaler_y.inverse_transform(np.vstack(y_true))
    y_pred = scaler_y.inverse_transform(np.vstack(y_pred))

    mse = np.mean((y_pred - y_true) ** 2)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Output: {output_name} | Test MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    # Gráfico de Paridade
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Parity Plot - {output_name}")
    plt.legend()
    plt.grid()
    plt.show()


# Função principal
def main():
    # Carregar os dados
    data = pd.read_csv('full_database_4000.csv')

    # Selecionar entradas e saídas
    input_columns = ["ALPHA", "BETA", "MACH", "ALTITUDE", "DF1", "DF2", "DF3", "DF4"]
    output_columns = ["CN", "CM", "CA", "CY", "CL", "CD", "CNA", "CMA", "CYB", "CLNB", "CLLB"]

    X = data[input_columns].values

    # Normalizar entradas
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    # Salvar scaler de entrada
    joblib.dump(scaler_X, "scaler_X.pkl")

    # Criar pasta para salvar modelos e scalers
    os.makedirs("models", exist_ok=True)

    for output in output_columns:
        print(f"\nTreinando modelo para saída: {output}")

        # Normalizar saída
        y = data[[output]].values
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)

        # Divisão entre treinamento, validação e teste
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Criar DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=32, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=32, shuffle=False)
        }

        # Inicializar o modelo
        input_size = len(input_columns)
        output_size = 1
        model = AerodynamicNN(input_size, output_size)

        # Configurar função de perda e otimizador
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Treinar o modelo
        model = train_model(model, dataloaders, criterion, optimizer, num_epochs=500, patience=50)

        # Avaliar o modelo
        evaluate_model(model, dataloaders['test'], scaler_y, output)

        # Salvar o modelo e scaler
        model_path = f"models/{output}_model_scripted.pt"
        scaler_path = f"models/{output}_scaler_y.pkl"
        scripted_model = torch.jit.script(model)
        scripted_model.save(model_path)
        joblib.dump(scaler_y, scaler_path)

        print(f"Modelo salvo: {model_path}")
        print(f"Scaler salvo: {scaler_path}")


# Executar o código principal
if __name__ == "__main__":
    main()