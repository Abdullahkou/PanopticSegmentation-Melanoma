from segmentation_models_pytorch import UnetPlusPlus
import torch.nn as nn
import ssl
import torch

# SSL-Konfiguration, um das Herunterladen der vortrainierten Gewichte ohne SSL-Fehler zu ermöglichen
ssl._create_default_https_context = ssl._create_unverified_context

# Definiere das Modell
class ForeUNetPlusPlus(nn.Module):
    def __init__(self, num_classes=2,in_channel=3):
        super(ForeUNetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel

        # U-Net++ Modell für die  nuclei-Segmentierung
        unetplusplus_tissue = UnetPlusPlus(
            encoder_name="densenet121",        # Verwende DenseNet121 als Encoder
            encoder_weights='imagenet',        # Vortrainierte Gewichte auf ImageNet
            decoder_use_batchnorm=True,        # Verwende BatchNorm im Decoder
            in_channels = self.in_channel,     #  
            classes=self.num_classes,          # Anzahl der Klassen
            activation=None,                   # Keine Aktivierungsfunktion am Ende
            encoder_depth=3,                   # Begrenze die Tiefe des Encoders
            decoder_channels=(128, 64, 32),     # Kanäle im Decoder
            decoder_attention_type="scse"  # Add attention (e.g., scSE)
        )
        
        #decoder_channels=(128, 64, 32),    # Kanäle im Decoder

        # Trenne Encoder, Decoder und Segmentierungskopf für Flexibilität
        self.encoder = unetplusplus_tissue.encoder
        self.decoder_tissue1 = unetplusplus_tissue.decoder
        self.seghead_tissue1 = unetplusplus_tissue.segmentation_head

    def forward(self, x):
        # Encoder-Pfad
        x = self.encoder(x)
        # Decoder-Pfad für Gewebe-Segmentierung
        decoder_tissue_out = self.decoder_tissue1(*x)
        # Erzeuge die Segmentierungsmaske
        mask_tissue = self.seghead_tissue1(decoder_tissue_out)

        return mask_tissue

# model = HalfDualDecUNetPlusPlus()
# print(model)
