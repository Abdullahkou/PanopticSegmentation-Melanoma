import pandas as pd
import albumentations as A
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import cv2
import ast
from tqdm import tqdm

class MTLDataset(data.Dataset):
    """
    PUMAPanopticDataset Class for processing histopathology image patches with both tissue and nuclei masks
    for panoptic segmentation as part of the PUMA Challenge.

    This version includes functions to convert color-coded masks into class-based masks.

    Attributes:
        patch_info (DataFrame): DataFrame with paths to image and both masks (nuclei and tissue).
        normalize (callable, optional): Function to normalize image data.
        do_color_augment (bool): Whether to apply color augmentation.
        aug_magnitude (int): Number of augmentations to apply at once.
        do_geometric_augment (bool): Whether to apply geometric augmentation.
        use_tissue_mask (bool): Flag to determine if tissue mask should be loaded.
        use_nuclei_mask (bool): Flag to determine if nuclei mask should be loaded.
        num_classes_tissue (int): Number of classes for the tissue segmentation task.
        num_classes_nuclei (int): Number of classes for the nuclei segmentation task.
    """

    def __init__(self,
                 patch_info,
                 add_gray_channel=False,
                 num_classes_tissue=4,
                 num_classes_nuclei=4):

        self.patch_info = patch_info
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.num_classes_tissue = num_classes_tissue
        self.num_classes_nuclei = num_classes_nuclei
        self.add_gray_channel = add_gray_channel



    def color_image_to_mask_nuc(self, color_image):
        color_to_class_map = {
        (255, 255, 255): 0,    # Background
        (200, 0, 0): 1,        # nuclei_tumor
        (255, 0, 255): 2,      # TILs (lymphocytes and plasma cells)
        (150, 200, 150): 3,    # Other cells (histiocytes, stromal cells, etc.)
        (0, 255, 0): 3,
        (51, 0, 51): 3,
        (0, 128, 128): 3,
        (204, 204, 51): 3,
        (102, 26, 51): 3,
        (51, 51, 51): 3
        }
    
        color_array = np.array(color_image)
        unique_colors = np.unique(color_array.reshape(-1, color_array.shape[2]), axis=0)
    
        # Erstelle eine leere Maske für die Klassenzuordnung
        mask = np.zeros((color_array.shape[0], color_array.shape[1]), dtype=np.uint8)
    
        for color in unique_colors:
            color_tuple = tuple(color)
            
            # Bestimme die Klasse, entweder durch explizite Zuordnung oder Standardklasse 3
            class_value = color_to_class_map.get(color_tuple, 3)
            
            is_color = np.all(color_array == color_tuple, axis=-1)
            
            if np.any(is_color):  # Nur wenn die Farbe im Bild vorkommt
                mask[is_color] = class_value
        return mask

    
    def color_image_to_mask(self, color_image):
        color_to_class_map = {
            (255, 255, 255): 0,
            (150, 200, 150): 1,
            (0, 255, 0): 2,
            (200, 0, 0): 3,
            (23, 172, 169): 2,
            (51, 0, 51): 2
        }
    
    
        color_array = np.array(color_image)
        mask = np.zeros((color_array.shape[0], color_array.shape[1]), dtype=np.uint8)
        
        for color, class_value in color_to_class_map.items():
            is_color = np.all(color_array == color, axis=-1)
            if np.any(is_color):  # Nur wenn die Farbe im Bild vorkommt
                mask[is_color] = class_value
                
        return mask
    

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        img_row = self.patch_info.iloc[idx]
        image_path = img_row['path']

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        n_image = self.normalize(image)  # Normalisiertes RGB-Bild als Tensor


        if self.add_gray_channel:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Graustufenkanal erzeugen
            gray_image = torch.tensor(gray_image, dtype=torch.float32) / 255.0  # Auf [0, 1] skalieren
            gray_image = gray_image.unsqueeze(0)  # Kanal-Dimension hinzufügen
        
            # RGB und Graustufenkanal kombinieren
            image = torch.cat((n_image, gray_image), dim=0)
        
        else:
            image = n_image

        return_values = [image]

        tissue_mask_path = img_row['tissue_mask']
        tissue_mask = Image.open(tissue_mask_path).convert('RGB')  # Load as RGB
        tissue_mask = self.color_image_to_mask(tissue_mask)  # Convert to class mask

        nuclei_mask_path = img_row['nuclei_mask']
        nuclei_mask = Image.open(nuclei_mask_path).convert('RGB')  # Load as RGB
        nuclei_mask = self.color_image_to_mask_nuc(nuclei_mask)  # Convert to class mask


        tissue_mask = torch.from_numpy(tissue_mask)
        tissue_mask_one_hot = np.eye(self.num_classes_tissue)[tissue_mask]  # One-hot encoding
        tissue_mask_one_hot = torch.from_numpy(tissue_mask_one_hot)
        return_values.append(tissue_mask_one_hot)

        nuclei_mask = torch.from_numpy(nuclei_mask)
        nuclei_mask_one_hot = np.eye(self.num_classes_nuclei)[nuclei_mask]  # One-hot encoding
        nuclei_mask_one_hot = torch.from_numpy(nuclei_mask_one_hot)
        return_values.append(nuclei_mask_one_hot)

        return tuple(return_values)

