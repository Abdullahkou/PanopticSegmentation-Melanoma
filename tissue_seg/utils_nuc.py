import torch
from pathlib import Path
import glob
import re
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import matplotlib.patches as mpatches
import seaborn as sns
from focal_loss import FocalLoss
import torch.nn.functional as F

def increment_path(path, exist_ok=False, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    # path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return Path(f"{path}{sep}{n}")  # update path


def f1_score(tp, fp, fn):
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F1 score
    return 2 * (precision * recall) / (precision + recall + epsilon)


# def loss(y_pred: torch.tensor, y_true: torch.tensor):
#     b, w, h, c = y_pred.size()
#     y_pred = y_pred.reshape(b * w * h, c)
#     # y_pred = torch.softmax(y_pred, dim=-1)
#     y_true = y_true.reshape(b * w * h, c)
#     loss_fn = nn.CrossEntropyLoss()
#     y_true = y_true.detach()
#     loss = loss_fn(y_pred.float(), y_true.float())
#     return loss




import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_classwise_centroids(masks, classes, num_classes):
    """ Berechnet Schwerpunkte pro Klasse. """
    b, w, h = masks.shape
    centroids = torch.zeros((b, num_classes, 2), device=masks.device)

    for c in range(1, num_classes):  # Klasse 0 ignorieren (Hintergrund)
        class_mask = (classes == c).float()
        if class_mask.sum() > 0:
            x_coords = torch.arange(w, device=masks.device).repeat(b, h, 1)
            y_coords = torch.arange(h, device=masks.device).repeat(b, w, 1).transpose(1, 2)

            centroids_x = torch.sum(class_mask * x_coords, dim=(1, 2)) / (torch.sum(class_mask, dim=(1, 2)) + 1e-7)
            centroids_y = torch.sum(class_mask * y_coords, dim=(1, 2)) / (torch.sum(class_mask, dim=(1, 2)) + 1e-7)

            centroids[:, c, 0] = centroids_x
            centroids[:, c, 1] = centroids_y

    return centroids

def focal_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: torch.Tensor = None, gamma: float = 2.0):
    """ Berechnet den Focal Loss mit Klassenpriorität. """
    if alpha is None:
        alpha = torch.tensor([0.1, 1.0, 2.0, 3.0])  # Default-Gewichte

    device = y_pred.device
    alpha = alpha.to(device)  

    b, w, h, c = y_pred.size()
    y_pred = y_pred.view(b * w * h, c)
    
    if y_true.dim() == 4:  # Falls One-Hot-Encoding
        y_true = y_true.argmax(dim=-1)
    y_true = y_true.view(b * w * h)

    # Focal Loss Berechnung
    log_prob = F.log_softmax(y_pred, dim=-1)
    prob = torch.exp(log_prob)
    
    focal_weight = torch.pow(1.0 - prob, gamma)
    ce_loss = F.nll_loss(log_prob, y_true.long(), weight=alpha, reduction='none')
    focal_loss = focal_weight.gather(1, y_true.view(-1, 1)).squeeze() * ce_loss

    return focal_loss.mean()

def combined_loss(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes=4):
    """ Kombinierte Loss-Funktion mit klassenabhängigem Centroid-Loss. """

    # Focal Loss (bleibt erhalten)
    focal = focal_loss(y_pred, y_true)

    # IoU-Loss (falls nötig, aber kaum gewichtet)
    y_pred_classes = y_pred.argmax(dim=-1)  
    y_pred_binary = (y_pred_classes > 0).float()
    y_true_binary = (y_true > 0).float()

    intersection = (y_pred_binary * y_true_binary).sum(dim=(1, 2))
    union = (y_pred_binary + y_true_binary).sum(dim=(1, 2)) - intersection
    iou_loss = (1 - (intersection / (union + 1e-7))).mean()

    # Berechnung des klassenabhängigen Centroid-Loss
    if y_true_binary.sum() > 0:
        y_pred_centroids = compute_classwise_centroids(y_pred_binary, y_pred_classes, num_classes)
        y_true_centroids = compute_classwise_centroids(y_true_binary, y_true, num_classes)

        centroid_loss = torch.mean((y_pred_centroids - y_true_centroids) ** 2)
    else:
        centroid_loss = 0

    # **Neue Gewichtung** – Centroid-Loss für richtige Klassifikation wichtiger!
#    total_loss = 0.1 * focal + 0.3 * iou_loss + 1.0 * centroid_loss
#    total_loss = 0.5 * focal + 0.2 * iou_loss + 5.0 * centroid_loss
    
    total_loss = 0.0 * focal + 0.2 * iou_loss + 1.0 * centroid_loss
    
    return total_loss

# Loss-Funktion für Training
def loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    return combined_loss(y_pred, y_true)



def focal_loss_old(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: torch.Tensor = torch.tensor([0.1, 1.0, 2.0, 3.0]), gamma: float = 2.0):
    """
    Verlustfunktion mit Focal Loss nach deinem vorgegebenen Muster.
    """
    device = y_pred.device
    class_weights = torch.tensor([0.5, 1.0, 2.0, 3.0]).to(device)  # Klassen-Gewichte

    # Tensoren reshapen
    b, w, h, c = y_pred.size()
    y_pred = y_pred.reshape(b * w * h, c)
    if y_true.dim() == 4:  # Falls One-Hot-Encoding verwendet wird
        y_true = y_true.argmax(dim=-1)
    y_true = y_true.reshape(b * w * h)

    # Focal Loss Berechnung
    gamma = 2.0  # Fokussierungsparameter
    log_prob = F.log_softmax(y_pred, dim=-1)  # Log-Wahrscheinlichkeiten
    prob = torch.exp(log_prob)  # Wahrscheinlichkeiten

    # Focal Gewichtung
    focal_weight = torch.pow(1.0 - prob, gamma)
    ce_loss = F.nll_loss(log_prob, y_true.long(), weight=class_weights, reduction='none')
    focal_loss = focal_weight.gather(1, y_true.view(-1, 1)).squeeze() * ce_loss

    return focal_loss.mean()


# def loss(y_pred: torch.tensor, y_true: torch.tensor):
#    return focal_loss(y_pred, y_true)




# def loss(y_pred: torch.tensor, y_true: torch.tensor):

    #device = y_pred.device
    #class_weights = torch.tensor([0.1, 1.0, 2.0, 3.0]).to(device)

    #b, w, h, c = y_pred.size()
    #y_pred = y_pred.reshape(b * w * h, c)
    #if y_true.dim() == 4:
    #    y_true = y_true.argmax(dim=-1)
        
    #y_true = y_true.reshape(b * w * h)

    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)    
    #loss = loss_fn(y_pred.float(), y_true.long())
    #return loss








def dice_loss(y_pred, y_true, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return 1 - dice

def combined_loss(y_pred: torch.Tensor, y_true: torch.Tensor, auxiliary_pred=None, auxiliary_true=None):

    a = 0.75
    b = 0.25
    
    # Dice and BCE Loss for segmentation
    b, w, h, c = y_pred.size()
    device = y_pred.device
    class_weights = torch.tensor([0.5, 1.0, 3.0, 7.0]).to(device)
    y_pred = y_pred.reshape(b * w * h, c)
    if y_true.dim() == 4:
        y_true = y_true.argmax(dim=-1)
    y_true = y_true.reshape(b * w * h)

    #Cross-Entropy Loss without class weights
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)    
    ce_loss = ce_loss_fn(y_pred.float(), y_true.long())

    #Dice Loss (apply separately for each class)
    dice_loss_value = 0
    for cls in range(0, c):
        dice_loss_value += dice_loss((y_pred[:, cls] == cls).float(), (y_true == cls).float())
    dice_loss_value /= (c-1)  # Average over classes

    # MSE Loss for auxiliary data (if provided)
    #mse_loss_value = F.mse_loss(auxiliary_pred, auxiliary_true) if auxiliary_pred is not None and auxiliary_true is not None else 0

    # Combine Losses
    total_loss = (a * ce_loss) + (b * dice_loss_value)
    return total_loss















class_map = {
    'Background': (255, 255, 255),        #Background
    'nuclei_tumor': (200, 0, 0),         # Tumor
    'nuclei_lymphocyte': (255, 0, 255),  # Lymphozyten (TILs)
    "Other cells": (0, 255, 0)             # other (stroma, neutrophil, endothelium usw.)
}




def plot_and_save_mask_comparison(x, y_true, y_pred, class_map, epoch, opt_name):
    """
    Plots and saves the original image, true mask, and predicted mask, and includes a legend.

    :param x: Tensor of the original image with shape [batch_size, height, width, channels]
    :param y_true: Tensor of true masks with shape [batch_size, height, width, num_classes]
    :param y_pred: Tensor of predicted masks with shape [batch_size, height, width, num_classes]
    :param class_map: Dictionary mapping class names to their respective colors
    :param epoch: used for a filename to save the plot
    """

    cmap_list = list(class_map.values())
    cmap = ListedColormap(np.array(cmap_list) / 255.0)

    # Select a random sample from the batch
    batch_size = y_true.shape[0]
    folder_path = os.path.join(
        r"train_preds/", opt_name)

    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for sample_index in range(0, batch_size):
        # sample_index = random.randint(0, batch_size - 1)

        # Extract the image, true mask, and predicted mask for the sample
        image_sample = x[sample_index].cpu().numpy()
        true_mask = y_true[sample_index].cpu().numpy()
        pred_mask = y_pred[sample_index].cpu().numpy()

        # Create a subplot
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot the original image
        axs[0].imshow(np.transpose(image_sample, (1, 2, 0)))
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # Plot the true mask
        axs[1].imshow(true_mask, cmap=cmap, vmin=0,
                      vmax=6, interpolation='nearest')
        axs[1].set_title('True Mask')
        axs[1].axis('off')

        # Plot the predicted mask
        axs[2].imshow(pred_mask, cmap=cmap, vmin=0,
                      vmax=6, interpolation='nearest')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        # Create a legend for the classes
        patches = [mpatches.Patch(color=np.array(color) / 255.0, label=class_name)
                   for class_name, color in class_map.items()]
        plt.legend(handles=patches, bbox_to_anchor=(
            1.05, 1), loc=2, borderaxespad=0.)

        # Now you can safely save files to this directory
        file_path = os.path.join(
            folder_path, f'pred_{epoch}_{sample_index}.png')

        # Save the plot
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()


def plot_cm(normalized_confusion_matrix):
    class_names = [
        "Background",          # Klasse 0
        "nuclei_tumor",        # Klasse 1
        "nuclei_lymphocyte",   # Klasse 2 (TILs: lymphocytes and plasma cells)
        "Other cells"          # Klasse 3 (histiocytes, stromal cells, etc.)
        ]


    plt.figure(figsize=(15, 15))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    return sns.heatmap(normalized_confusion_matrix, annot=True, cmap='viridis',
                       vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names).get_figure()
