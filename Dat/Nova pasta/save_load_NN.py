# Converter o modelo treinado para TorchScript
scripted_model = torch.jit.script(model)

# Salvar o modelo TorchScript
scripted_model.save("model_scripted.pt")
print("TorchScript model saved as 'model_scripted.pt'")



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



#JULIA

import torch
import bson
from bson import dumps

# Configuração do modelo
model_config = {
    "input_size": len(input_columns),
    "output_size": len(output_columns),
    "hidden_layers": [128, 128],  # Arquitetura da rede
}

# Exportar os pesos do modelo
model_weights = model.state_dict()

# Exportar os escaladores
scalers = {
    "scaler_X": scaler_X.mean_.tolist(),  # Média das entradas
    "scaler_X_scale": scaler_X.scale_.tolist(),  # Escala das entradas
    "scaler_y": scaler_y.mean_.tolist(),  # Média das saídas
    "scaler_y_scale": scaler_y.scale_.tolist(),  # Escala das saídas
}

# Salvar tudo em BSON
export_data = {
    "model_config": model_config,
    "model_weights": {k: v.tolist() for k, v in model_weights.items()},
    "scalers": scalers,
}

# Escrever em arquivo BSON
with open("model_data.bson", "wb") as f:
    f.write(dumps(export_data))
print("Model exported as BSON (model_data.bson).")