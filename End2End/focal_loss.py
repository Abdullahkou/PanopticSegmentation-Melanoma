import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Setze alpha falls angegeben, um eine Gewichtung für die Klassen festzulegen
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1 - alpha])
            elif isinstance(alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(alpha)
            else:
                raise ValueError("Alpha muss ein float, int oder eine Liste sein.")
        else:
            self.alpha = None

    def forward(self, inputs, targets):


        CE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        pt = torch.exp(-CE_loss)
        
        # Alpha-Gewichtung anwenden, falls vorhanden
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            logpt = CE_loss * at
        else:
            logpt = CE_loss

        # Focal Loss Berechnung
        focal_loss = -((1 - pt) ** self.gamma) * logpt

        # Reduktion des Verlusts
        if self.reduction == 'mean':
            result = focal_loss.mean()
        elif self.reduction == 'sum':
            result = focal_loss.sum()
        else:  # 'none'
            result = focal_loss
        
        return result







