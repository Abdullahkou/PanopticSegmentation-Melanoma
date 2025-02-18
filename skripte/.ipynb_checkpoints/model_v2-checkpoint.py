import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetPlusPlusMultiTask(smp.UnetPlusPlus):
    def __init__(self, encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet',
                 decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None, in_channels=3, classes=1, activation=None,
                 aux_params=None, num_classes_tissue=5, num_classes_nuclei=3):
        super().__init__(encoder_name=encoder_name, encoder_depth=encoder_depth,
                         encoder_weights=encoder_weights, decoder_use_batchnorm=decoder_use_batchnorm,
                         decoder_channels=decoder_channels, decoder_attention_type=decoder_attention_type,
                         in_channels=in_channels, classes=num_classes_tissue, activation=activation,
                         aux_params=aux_params)

        # Zusätzliche Segmentation Head für Zellkernerkennung
        self.segmentation_head_nuclei = smp.base.SegmentationHead(
            in_channels=self.decoder.out_channels[-1],
            out_channels=num_classes_nuclei,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x):
        """Vorwärtsdurchlauf des Modells.

        Args:
            x: Eingabebildtensor der Form (N, C, H, W).

        Returns:
            tissue_output: Ausgabe für die Gewebesegmentierung.
            nuclei_output: Ausgabe für die Zellkernerkennung.
        """
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        tissue_output = self.segmentation_head(decoder_output)
        nuclei_output = self.segmentation_head_nuclei(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return tissue_output, nuclei_output, labels

        return tissue_output, nuclei_output
