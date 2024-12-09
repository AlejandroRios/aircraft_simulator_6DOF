import torch
import torch.nn as nn
import numpy as np

# Carregar o modelo TorchScript
scripted_model = torch.jit.load("model_scripted.pt")
scripted_model.eval()  # Certifique-se de que está em modo de avaliação

# Preparar os dados de entrada
input_data = np.array([[5.0, 0.0, 0.8, 1000, 0.0, 0.0, 0.0, 0.0]])  # Exemplo de entrada
input_normalized = scaler_X.transform(input_data)  # Normalizar os dados

# Fazer a previsão
input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
output_normalized = scripted_model(input_tensor).detach().numpy()
output = scaler_y.inverse_transform(output_normalized)  # Reverter a normalização

print("Predicted Outputs:", output)