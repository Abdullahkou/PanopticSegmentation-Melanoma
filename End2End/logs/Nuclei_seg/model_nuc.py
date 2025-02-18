from segmentation_models_pytorch import UnetPlusPlus
import torch.nn as nn
import ssl
import torch

ssl._create_default_https_context = ssl._create_unverified_context

class HalfDualDecUNetPlusPlus(nn.Module):
    def __init__(self, num_classes=4, in_channels=3):
        super(HalfDualDecUNetPlusPlus, self).__init__()
        self.num_classes = num_classes

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

        self.unetplusplus2 = UnetPlusPlus(
            encoder_name="densenet121",
            encoder_weights="imagenet",
            decoder_use_batchnorm=True,
            in_channels=num_classes + in_channels,  # RGB + Logits des ersten Modells
            classes=num_classes,
            activation=None,
            encoder_depth=3,
            decoder_channels=(128, 64, 32),
            decoder_attention_type="scse"
        )
#            in_channels=num_classes + in_channels -1,  # RGB + Logits des ersten Modells wenn gry weg muss

    def forward(self, x):
        #print(x.shape)
        #x = x.permute(0, 3, 1, 2)  # (Batch, Höhe, Breite, Kanäle) -> (Batch, Kanäle, Höhe, Breite)
        logits1 = self.unetplusplus1(x)

        #logits1_resized = nn.functional.interpolate(logits1, size=x.shape[2:], mode='bilinear', align_corners=False)
        #logits1i = logits1.clone()  # Klone den Tensor, bevor er erneut verwendet wird
        #x = x[:, :-1, :, :]  ## gray weg 
        
        combined_input = torch.cat([x, logits1], dim=1)

        # Zweites U-Net++ (finale Segmentierung)
        logits2 = self.unetplusplus2(combined_input)

        return logits1, logits2

# model = DualUNetPlusPlus(num_classes=4, in_channels=3)
# print(model)
