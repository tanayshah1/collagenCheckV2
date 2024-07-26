import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50),  # Ensure this matches the number of features
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(25, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim),  # Output should match input_dim
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
