"""
Model Architecture Module
-------------------------
Defines the sequence-based model architecture used for clinical time-series reconstruction.
"""
import torch
import torch.nn as nn

class ClinicalAutoencoder(nn.Module):
    """
    Autoencoder for clinical time-series data.
    
    Attributes:
        encoder (nn.Sequential): Compresses input data to latent representation.
        decoder (nn.Sequential): Reconstructs data from latent representation.
    """
    def __init__(self, input_dim: int):
        super(ClinicalAutoencoder, self).__init__()
        
        # Encoder: Compresses input features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Decoder: Attempts to reconstruct original input
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        return self.decoder(self.encoder(x))