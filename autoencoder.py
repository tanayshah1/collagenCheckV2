import torch
import torch.nn as nn

class RatioAutoencoder(nn.Module):
    def __init__(self):
        super(RatioAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 5),  # Adjust input and output dimensions as needed
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),  # Adjust input and output dimensions as needed
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
