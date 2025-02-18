from segmentation_models_pytorch import UnetPlusPlus
import segmentation_models_pytorch as smp
import torch.nn as nn
import ssl
import torch
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

# SSL-Konfiguration, um das Herunterladen der vortrainierten Gewichte ohne SSL-Fehler zu ermöglichen
ssl._create_default_https_context = ssl._create_unverified_context

# Definiere das Modell
class MultiTaskUnetPlusPlus(nn.Module):
    def __init__(self, num_classes_task1_tissue=6, num_classes_task2_nuclei=4, in_channels=3, encoder_depth=3):
        super(MultiTaskUnetPlusPlus, self).__init__()
        
        # Number of classes for each segmentation task
        self.num_classes_task1 = num_classes_task1_tissue
        self.num_classes_task2 = num_classes_task2_nuclei
        
        self.in_channels = in_channels
        self.encoder_depth = encoder_depth
        # Gemeinsamer Encoder
        self.base_model = UnetPlusPlus(
            encoder_name="densenet121",        # Verwende DenseNet121 als Encoder
            encoder_weights='imagenet',        # Vortrainierte Gewichte auf ImageNet
            decoder_use_batchnorm=True,        # Verwende BatchNorm im Decoder
            in_channels=self.in_channels,                     # RGB-Bilder
            classes=self.num_classes_task1,    # Anzahl der Klassen (Dummy-Wert hier)
            activation=None,                   # Keine Aktivierungsfunktion am Ende
            encoder_depth=self.encoder_depth,                   # Begrenze die Tiefe des Encoders
            decoder_channels=(256, 128, 64,32) # Kanäle im Decoder
        )

        # Encoder übernehmen (gemeinsamer Encoder)
        self.encoder = self.base_model.encoder

        # Eigene Decoder für Task A und Task B
        self.decoder_task1 = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32),
            n_blocks=self.encoder_depth,
            use_batchnorm=True,
            center=False,
            attention_type="scse"
        )
        self.seghead_task1 = nn.Conv2d(32, self.num_classes_task1, kernel_size=1)

        self.decoder_task2 = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32),
            n_blocks=self.encoder_depth,
            use_batchnorm=True,
            center=False,
            attention_type="scse"
        )
        self.seghead_task2 = nn.Conv2d(32, self.num_classes_task2, kernel_size=1)

    def forward(self, x):
        # Gemeinsame Encoder-Ausgabe
        encoder_outputs = self.encoder(x)

        # Task 1 tissue
        decoder_task1_out = self.decoder_task1(*encoder_outputs)
        tissue_mask = self.seghead_task1(decoder_task1_out)

        # Task 2 nuclei
        decoder_task2_out = self.decoder_task2(*encoder_outputs)
        nuclei_mask = self.seghead_task2(decoder_task2_out)

        return tissue_mask, nuclei_mask




# model = MultiTaskUnetPlusPlus()
# print(model)
