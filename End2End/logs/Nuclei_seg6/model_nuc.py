from segmentation_models_pytorch import UnetPlusPlus
import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)

class IndependentUnetPlusPlus1(nn.Module):
    def __init__(self, num_classes=4, in_channels=3):
        super(IndependentUnetPlusPlus1, self).__init__()
        self.unetplusplus1 = UnetPlusPlus(
            encoder_name="densenet121",
            encoder_weights="imagenet",
            decoder_use_batchnorm=True,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            encoder_depth=3,
            decoder_channels=(128, 64, 32),
            decoder_attention_type="scse"
        )

    def forward(self, x):
        logits1 = self.unetplusplus1(x)
        return logits1


class IndependentUnetPlusPlus2(nn.Module):
    def __init__(self, num_classes=4, in_channels=7):  # 3 RGB + 4 Logits von Netz 1
        super(IndependentUnetPlusPlus2, self).__init__()
        self.unetplusplus2 = UnetPlusPlus(
            encoder_name="densenet121",
            encoder_weights="imagenet",
            decoder_use_batchnorm=True,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            encoder_depth=3,
            decoder_channels=(128, 64, 32),
            decoder_attention_type="scse"
        )

    def forward(self, x):
        logits2 = self.unetplusplus2(x)
        return logits2


# Hauptklasse für Training und Verbindung der beiden Netzwerke
class HalfDualDecUNetPlusPlus(nn.Module):
    def __init__(self, num_classes=4, in_channels=3):
        super(HalfDualDecUNetPlusPlus, self).__init__()
        self.net1 = IndependentUnetPlusPlus1(num_classes=num_classes, in_channels=in_channels)
        self.net2 = IndependentUnetPlusPlus2(num_classes=num_classes, in_channels=num_classes + in_channels)

    def forward(self, x):
        # Erstes Netz
        logits1 = self.net1(x)

        # Kombiniere Eingabe mit den Logits des ersten Netzes
        combined_input = torch.cat([x.clone(), logits1.clone()], dim=1)
        
        # Zweites Netz
        logits2 = self.net2(combined_input)

        return logits1, logits2




