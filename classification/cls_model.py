import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import ssl
import segmentation_models_pytorch as smp
import torch.nn.functional as F

ssl._create_default_https_context = ssl._create_unverified_context


class UNetPlusPlusClassifier(nn.Module):
    def __init__(self, encoder_name="densenet121", encoder_weights="imagenet", num_classes=3):
        super(UNetPlusPlusClassifier, self).__init__()
        self.unetpp = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=0,  # Kein direktes Segmentierungs-Output
            in_channels=4,
            activation=None,
            encoder_depth=2,  # Weniger Tiefe, bessere Konvergenz
            decoder_channels=(64, 32),  # Kleinere Decoder-Schichten
        )
        self.encoder = self.unetpp.encoder
        feature_dim = self.encoder.out_channels[-1]  # Letzte Feature-Map-Größe

        # Adaptive Pooling zur Reduktion auf 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers mit BatchNorm und Dropout
        self.fc1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, num_classes)  # Finale Klassifikation

    def forward(self, x):
        features = self.encoder(x)
        x_feat = self.pool(features[-1])  # Letztes Feature-Map nehmen und poolen
        x_feat = torch.flatten(x_feat, 1)  # Flatten für FC-Layer
        
        # Fully Connected Schichten mit BatchNorm und Dropout
        x = F.relu(self.bn1(self.fc1(x_feat)))
        x = self.fc2(x)  # Finale Logits
        return x


#model = UNetPlusPlusClassifier()
# Optional: Ausgabe der Modellstruktur
#print(model)

