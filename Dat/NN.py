import numpy as np
import pandas as pd
from pyDOE2 import lhs

def generate_doe_pydoe(n_samples, bounds):
    """
    Generate a Design of Experiments using Latin Hypercube Sampling (LHS).
    
    Parameters:
    - n_samples: Number of samples to generate.
    - bounds: List of (lower, upper) bounds for each parameter [(lb1, ub1), ...].
    
    Returns:
    - A pandas DataFrame with the sampled values.
    """
    n_inputs = len(bounds)
    X = lhs(n_inputs, samples=n_samples)

    # Scale the samples to the desired bounds
    scaled_X = np.zeros_like(X)
    for i, (lb, ub) in enumerate(bounds):
        scaled_X[:, i] = lb + (ub - lb) * X[:, i]
    
    columns = ["MACH", "ALTITUDE", "ALPHA"]
    return pd.DataFrame(scaled_X, columns=columns)

# Define the bounds
bounds = [
    (0.5, 2.5),      # MACH
    (0, 20000),      # ALTITUDE
    (-5, 20)         # ALPHA
]

# Generate the DOE
doe_data = generate_doe_pydoe(n_samples=100, bounds=bounds)
print(doe_data.head())


import os
from input_generator import MissileDATCOMInput
from output_reader import DATCOMResultReader

def run_datcom_for_conditions(conditions_df, datcom_input_path="for005.dat", datcom_output_path="for006.dat"):
    results = []

    for _, row in conditions_df.iterrows():
        # Generate DATCOM input
        datcom_input = MissileDATCOMInput(
            mach_vals=[row['MACH']],
            alt_vals=[row['ALTITUDE']],
            alpha_vals=[row['ALPHA']]
        )
        datcom_input.gerar_input(datcom_input_path)

        # Run DATCOM (adjust for your system command)
        os.system(f"missile_datcom.exe < {datcom_input_path}")

        # Read DATCOM output
        reader = DATCOMResultReader(datcom_output_path)
        reader.read_file()

        # Collect results
        dataframes = reader.get_dataframes()
        for dtype, df in dataframes:
            if dtype == "LONGITUDINAL":
                for i in range(len(df)):
                    results.append({
                        "MACH": row["MACH"],
                        "ALTITUDE": row["ALTITUDE"],
                        "ALPHA": row["ALPHA"],
                        "CL": float(df["CL"].iloc[i]),
                        "CD": float(df["CD"].iloc[i])
                    })

    return pd.DataFrame(results)

# Run DATCOM for the DOE
results_df = run_datcom_for_conditions(doe_data)
print(results_df.head())


import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_neural_network(data):
    # Prepare input and output data
    X = data[["MACH", "ALTITUDE", "ALPHA"]].values
    y = data[["CL", "CD"]].values  # Only `CL` and `CD` are used

    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(3, 64),  # 3 inputs: MACH, ALTITUDE, ALPHA
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # 2 outputs: CL, CD
            )

        def forward(self, x):
            return self.layers(x)

    # Initialize model, loss, and optimizer
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validate the model
        model.eval()
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    return model, scaler_X, scaler_y

    # Initialize model, loss, and optimizer
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Train the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validate the model
        model.eval()
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    return model, scaler_X, scaler_y

# Train the model
model, scaler_X, scaler_y = train_neural_network(results_df)


def predict_coefficients(model, scaler_X, scaler_y, mach, altitude, alpha):
    input_data = [[mach, altitude, alpha]]
    input_scaled = scaler_X.transform(input_data)
    prediction = model(torch.tensor(input_scaled, dtype=torch.float32))
    return scaler_y.inverse_transform(prediction.detach().numpy())

# Example prediction
predicted_coefficients = predict_coefficients(model, scaler_X, scaler_y, 2.0, 10000, 5)
print("Predicted Coefficients (CL, CD):", predicted_coefficients)

