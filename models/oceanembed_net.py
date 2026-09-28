import torch
import torch.nn as nn

class OceanEmbedNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        super(OceanEmbedNet, self).__init__()
        
        # Encoder: Compresses surface variables into a latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs subsurface depth layers
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstruction = self.decoder(latent_features)
        
        return reconstruction, latent_features