# Clinical MLOps: Time-Series Anomaly Detection Pipeline 🏥⚕️

An end-to-end Machine Learning Operations (MLOps) architecture designed to train, track, and serve a PyTorch Autoencoder for detecting anomalies in sparse clinical time-series data (Electronic Health Records).

## 🚀 Enterprise Architecture Overview
This repository demonstrates a production-ready ML lifecycle, moving beyond static notebooks into a fully containerized microservice architecture. 

* **Data Engineering:** Simulates and preprocesses sparse, irregularly sampled patient trajectories, handling informative missingness via forward-filling imputation.
* **Model Training:** PyTorch-based neural network autoencoder.
* **Experiment Tracking:** MLflow integration for automated hyperparameter logging and model registry.
* **Model Serving:** High-performance REST API built with FastAPI and Pydantic for real-time inference.
* **Orchestration:** Docker Compose network linking the training environment, tracking server, and inference API.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AhadAshfaq/clinical-mlops-pipeline.git
   cd clinical-mlops-pipeline