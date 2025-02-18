from segmentation_models_pytorch import Unet
import torch.nn as nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class HalfDualDecUNetPP(nn.Module):
    def __init__(self, num_classes=6):
        super(HalfDualDecUNetPP, self).__init__()
        self.num_classes = num_classes
        unet_tissue = Unet(
            encoder_name="densenet121",
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            in_channels=3,
            classes=self.num_classes,
            activation=None, encoder_depth=3, decoder_channels=(128, 64, 32))

        encoder = unet_tissue.encoder
        decoder_tissue = unet_tissue.decoder
        seghead_tissue = unet_tissue.segmentation_head

        self.encoder = encoder
        self.decoder_tissue1 = decoder_tissue
        self.seghead_tissue1 = seghead_tissue

    def forward(self, x):
        x = self.encoder(x)
        decoder_tissue_out = self.decoder_tissue1(*x)
        mask_tissue = self.seghead_tissue1(decoder_tissue_out)

        return mask_tissue
