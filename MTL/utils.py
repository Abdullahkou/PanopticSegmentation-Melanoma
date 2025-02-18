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

def loss(y_pred: torch.tensor, y_true: torch.tensor):
    
    class_weights = torch.tensor([0, 34, 41, 10, 87, 305], dtype=torch.float32)
    
    # Verschiebe die class_weights auf das gleiche Gerät wie y_pred
    device = y_pred.device
    class_weights = class_weights.to(device)

    b, w, h, c = y_pred.size()
    y_pred = y_pred.reshape(b * w * h, c)
    if y_true.dim() == 4:
        y_true = y_true.argmax(dim=-1)
        
    y_true = y_true.reshape(b * w * h)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights,ignore_index=0)
    
    loss = loss_fn(y_pred.float(), y_true.long())
    return loss


# def loss(y_pred: torch.tensor, y_true: torch.tensor):
#     # loss_fn = FocalLoss(ignore_index=8)  # Adjust the ignore_index as necessary
#     # loss = loss_fn(y_pred, y_true)

#     alpha_weights = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0])
#     loss_fn = FocalLoss(alpha=alpha_weights, gamma=2.0, ignore_index=8)
#     return loss_fn(y_pred, y_true)


class_map = {
    "Tissue White Background": (255, 255, 255),     # tissue_white_background0
    "Tissue Stroma": (150, 200, 150),               # tissue_stroma1
    "Tissue Blood Vessel": (0, 255, 0),             # tissue_blood_vessel2
    "Tissue Tumor": (200, 0, 0),                    # tissue_tumor3
    "Tissue Epidermis": (99, 145, 164),             # tissue_epidermis4
    "Tissue Necrosis": (51, 0, 51)                  # tissue_necrosis5
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
        "D:\\Code\\Artefact\\artefact\\images\\train", opt_name)

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
    "Tissue White Background",  
    "Tissue Stroma",
    "Tissue Blood Vessel",
    "Tissue Tumor",
    "Tissue Epidermis",
    "Tissue Necrosis"
    ]

    plt.figure(figsize=(15, 15))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    return sns.heatmap(normalized_confusion_matrix, annot=True, cmap='viridis',
                       vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names).get_figure()
