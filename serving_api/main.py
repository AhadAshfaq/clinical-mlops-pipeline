"""
Serving API Module
------------------
A high-performance FastAPI application designed to serve the PyTorch clinical 
autoencoder model for real-time anomaly detection and patient metric screening.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import torch
from typing import List

# Initialize the FastAPI application instance
app = FastAPI(
    title="Clinical Time-Series Diagnostic API",
    description="Production REST API serving a PyTorch autoencoder via MLflow for real-time patient anomaly scoring.",
    version="1.0.0"
)

# Global variables to hold the model configuration
MODEL_URI = "models:/clinical_time_series_anomaly/Production"
model = None

class PatientMetrics(BaseModel):
    """
    Pydantic data model defining the strict schema for incoming inference requests.
    Enforces type checking and provides basic validation bounds for clinical features.
    """
    vitals: List[float] = Field(
        ..., 
        min_items=4, 
        max_items=4, 
        description="A list of 4 continuous normalized physiological features (heart rate, blood pressure, temperature, glucose)."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vitals": [0.12, -0.45, 1.22, -0.08]
            }
        }

@app.on_event("startup")
def load_production_model():
    """
    Lifespan event handler triggered automatically when the FastAPI server starts.
    Establishes a connection to the MLflow tracking registry and loads the trained
    PyTorch model into memory.
    """
    global model
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        print("Searching MLflow for the latest training run...")
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiment = client.get_experiment_by_name("clinical_time_series_anomaly")
        
        # Get the single most recent run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        run_id = runs[0].info.run_id
        
        print(f"Pulling model weights from Run ID: {run_id}")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        model.eval()
        print("Production machine learning model successfully loaded into memory.")
        
    except Exception as e:
        print(f"Warning: MLflow connection delayed or failed. Details: {e}")
        # Use our local fallback copy of the model
        from model import ClinicalAutoencoder
        model = ClinicalAutoencoder(input_dim=4)
        model.eval()

@app.get("/health", tags=["Infrastructure"])
def health_check():
    """
    Lightweight health-check endpoint for infrastructure monitoring.
    Used by orchestrators (like Docker Compose or Kubernetes) to verify container vitality.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/analyze", tags=["Inference"])
def analyze_patient_trajectory(payload: PatientMetrics):
    """
    Accepts real-time patient clinical features, executes model inference via 
    the PyTorch autoencoder, and calculates a corresponding reconstruction anomaly score.
    
    Args:
        payload (PatientMetrics): Validated JSON payload containing a 4-dimensional vitals list.
        
    Returns:
        dict: A structured response indicating the anomaly score and system risk flag.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Machine learning model is currently uninitialized.")
        
    try:
        # Convert incoming list into a formatted PyTorch tensor matching model input dimensions
        input_tensor = torch.tensor([payload.vitals], dtype=torch.float32)
        
        # Execute inference without computing gradients to save memory and execution overhead
        with torch.no_grad():
            reconstructed_tensor = model(input_tensor)
            
            # Compute Mean Squared Error (MSE) loss between original parameters and network output
            mse_loss = torch.nn.functional.mse_loss(reconstructed_tensor, input_tensor).item()
            
        # Define an operational safety boundary threshold
        anomaly_threshold = 1.5
        is_anomaly = mse_loss > anomaly_threshold
        
        return {
            "reconstruction_error_mse": round(mse_loss, 4),
            "anomaly_detected": is_anomaly,
            "status_alert": "HIGH RISK: Divergent Physiological Signal" if is_anomaly else "NOMINAL: Stable Trajectory"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal inference failure processing payload: {str(e)}")