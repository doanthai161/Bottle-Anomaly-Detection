import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),)
        self.decoder = nn.Sequential( nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
