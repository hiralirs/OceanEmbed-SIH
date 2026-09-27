import torch
import torch.nn as nn

class OceanEmbedNet(nn.Module):
    def __init__(self):
        super(OceanEmbedNet, self).__init__()
        
        # Encoder: Compresses 5 surface variables into a latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs 15 subsurface depth layers (0m to 1000m)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 15, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent_features = self.encoder(x)
        reconstruction = self.decoder(latent_features)
        
        return reconstruction, latent_features