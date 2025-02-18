import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Gewichtungen der Klassen
        self.gamma = gamma  # Fokussierungsparameter
        self.reduction = reduction  # 'mean' oder 'sum'

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # pt ist die vorhergesagte Wahrscheinlichkeit
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Berechnung des Dice Loss
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        total = (inputs + targets_one_hot).sum(dim=(2, 3))
        dice_score = (2.0 * intersection + smooth) / (total + smooth)
        dice_loss = 1 - dice_score
        return dice_loss.mean()

class MultiTaskFocalDiceLoss(nn.Module):
    def __init__(self, alpha_tissue=None, alpha_nuclei=None, gamma=2.0, dice_weight=0.5, alpha=0.5, beta=0.5):
        super(MultiTaskFocalDiceLoss, self).__init__()
        self.alpha = alpha  # Gewicht für Gewebesegmentierung
        self.beta = beta    # Gewicht für Zellkernerkennung
        self.gamma = gamma
        self.dice_weight = dice_weight  # Gewichtung zwischen Dice und Focal Loss

        # Focal Loss für die Zellkernerkennung
        self.focal_loss_nuclei = FocalLoss(alpha=alpha_nuclei, gamma=gamma)
        
        # Cross-Entropy Loss und Dice Loss für Gewebesegmentierung
        self.ce_loss_tissue = nn.CrossEntropyLoss(weight=alpha_tissue)
        self.dice_loss_tissue = DiceLoss()  # Verwende DiceLoss ohne alpha_tissue

    def forward(self, tissue_output, nuclei_output, tissue_target, nuclei_target):
        # Gewebe-Segmentierungs-Verlust (Cross Entropy + Dice)
        ce_loss_tissue = self.ce_loss_tissue(tissue_output, tissue_target)
        dice_loss_tissue = self.dice_loss_tissue(tissue_output, tissue_target)
        
        total_tissue_loss = self.dice_weight * dice_loss_tissue + (1 - self.dice_weight) * ce_loss_tissue
        
        # Zellkern-Erkennungs-Verlust (Focal Loss)
        focal_loss_nuclei = self.focal_loss_nuclei(nuclei_output, nuclei_target)
        
        # Kombinierter Verlust
        total_loss = self.alpha * total_tissue_loss + self.beta * focal_loss_nuclei
        return total_loss