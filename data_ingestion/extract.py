"""
Data Ingestion Module
---------------------
Simulates the extraction and preprocessing of sparse, irregularly sampled
clinical Electronic Health Records (EHR) time-series data.
"""
import pandas as pd
import numpy as np
import os

def generate_clinical_data(num_patients: int = 100, max_time_steps: int = 48) -> pd.DataFrame:
    """
    Generates simulated sparse clinical data (e.g., ICU vitals).
    
    Args:
        num_patients (int): Number of unique patient trajectories to simulate.
        max_time_steps (int): Maximum hours of monitoring per patient.
        
    Returns:
        pd.DataFrame: A formatted pandas DataFrame containing the simulated EHR data.
    """
    print(f"Extracting data for {num_patients} patients...")
    records = []
    
    for patient_id in range(num_patients):
        # Not all patients stay the full 48 hours
        stay_length = np.random.randint(12, max_time_steps)
        
        for hour in range(stay_length):
            # Simulate sparse measurements (informational missingness)
            # Vitals are not measured every single hour, leading to NaNs
            record = {
                "patient_id": patient_id,
                "time_step": hour,
                "heart_rate": np.random.normal(80, 15) if np.random.rand() > 0.3 else np.nan,
                "blood_pressure": np.random.normal(120, 20) if np.random.rand() > 0.5 else np.nan,
                "temperature": np.random.normal(37.0, 0.5) if np.random.rand() > 0.7 else np.nan,
                "glucose_level": np.random.normal(100, 25) if np.random.rand() > 0.8 else np.nan
            }
            records.append(record)
            
    df = pd.DataFrame(records)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the extracted data by handling missing values via forward-filling,
    which is a standard baseline practice for clinical time-series.
    
    Args:
        df (pd.DataFrame): The raw, sparse clinical data.
        
    Returns:
        pd.DataFrame: The cleaned, dense data ready for model training.
    """
    print("Preprocessing sparse clinical trajectories...")
    
    # Sort chronologically per patient
    df = df.sort_values(by=["patient_id", "time_step"])
    
    # Forward-fill missing values per patient trajectory
    df = df.groupby("patient_id").ffill()
    
    # Fill any remaining NaNs at the start of trajectories with population medians
    df = df.fillna(df.median())
    
    return df

def save_data(df: pd.DataFrame, output_dir: str = "../data") -> None:
    """
    Saves the processed data to a designated directory for the training pipeline.
    
    Args:
        df (pd.DataFrame): The clean clinical data.
        output_dir (str): The target directory for the output file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_path = os.path.join(output_dir, "processed_clinical_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Data successfully saved to {file_path}")

if __name__ == "__main__":
    # Execute the Data Ingestion pipeline
    np.random.seed(42) # For reproducibility
    raw_data = generate_clinical_data(num_patients=500)
    clean_data = preprocess_data(raw_data)
    save_data(clean_data)
    print("Data ingestion pipeline completed successfully.")