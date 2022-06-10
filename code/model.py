import torch
import torch.nn as nn
from torchvision import datasets, models, transforms, utils

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        # import model
        self.model = models.resnet50(pretrained= True)

        # modify output network channel
        self.model.fc = nn.Linear(2048, 6)

    def forward(self, x):
        logits = self.model(x)
        return logits

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()

        # import model
        self.model = models.resnet101(pretrained= True)

        # modify output network channel
        self.model.fc = nn.Linear(2048, 5)

    def forward(self, x):
        logits = self.model(x)
        return logits

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        # import model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),        
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),   # medium: remove this layer
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1), # medium: remove this layer
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out