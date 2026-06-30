"""
Training Pipeline Module
------------------------
Handles data loading, training iterations, and MLflow experiment logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import pandas as pd
import numpy as np
from model import ClinicalAutoencoder

def load_clinical_data(file_path: str):
    """Loads the preprocessed CSV data and converts it to PyTorch tensors."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # We drop patient_id and time_step, keeping only the 4 clinical vitals
    features = df[['heart_rate', 'blood_pressure', 'temperature', 'glucose_level']].values
    
    # Normalize the data (Standard Scaling)
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    # Convert to PyTorch tensors
    tensor_x = torch.tensor(features, dtype=torch.float32)
    return TensorDataset(tensor_x, tensor_x) # Autoencoders use the input as the target

def train_model():
    """Executes the training loop and logs metrics to MLflow."""
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("clinical_time_series_anomaly")

    # Load real data
    dataset = load_clinical_data("/app/data/processed_clinical_data.csv")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    with mlflow.start_run():
        # Configuration
        input_dim = 4 # 4 vital signs
        epochs = 10
        model = ClinicalAutoencoder(input_dim=input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 64)

        # 2. Real Training Loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # 3. Log the Final Model to MLflow Registry
        # Create a dummy input tensor with a batch size > 1 to prevent PyTorch from hardcoding the dimension
        example_input = np.random.randn(8, input_dim).astype(np.float32)
        mlflow.pytorch.log_model(model, "model", input_example=example_input)
        
        print("Training complete. Model logged to MLflow Registry.")

if __name__ == "__main__":
    train_model()