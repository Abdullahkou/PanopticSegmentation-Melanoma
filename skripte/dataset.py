import albumentations as A
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import os
from he_randaugment import randaugment


class TissueSegmentationDataset(data.Dataset):
    """
    TissueSegmentationDataset Class for processing histopathology image patches with tissue masks only.

    Attributes:
        patch_info (DataFrame): DataFrame with paths to image and tissue mask.
        normalize (callable, optional): Function to normalize image data.
        do_color_augment (bool): Whether to apply color augmentation.
        aug_magnitude (int): Number of augmentations to apply at once.
        do_geometric_augment (bool): Whether to apply geometric augmentation.
        num_classes_tissue (int): Number of classes for the tissue segmentation task.
    """

    def __init__(self,
                 patch_info,
                 normalize=None,
                 geometric_augment: bool = True,
                 color_augment: bool = True,
                 augment_magnitude: int = 3,
                 num_classes_tissue=6):  # Anzahl der Tissue-Klassen

        self.patch_info = patch_info
        self.normalize = normalize
        self.do_color_augment = color_augment
        self.aug_magnitude = augment_magnitude
        self.do_geometric_augment = geometric_augment
        self.num_classes_tissue = num_classes_tissue

        # Define geometric augmentations
        aug_targets = {'tissue_mask': 'mask'}

        self.geoaug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ], additional_targets=aug_targets, p=0.8)

        # Define color augmentations
        self.coloraug = A.SomeOf([
            A.Blur(blur_limit=5, p=0.4),
            A.RandomBrightnessContrast(p=0.5)
        ], n=augment_magnitude, p=0.6)

    # Function to convert tissue color-coded masks to class-based masks
    def color_image_to_mask(self, color_image):
        color_to_class_map = {
            (255, 255, 255): 0,  # Hintergrund
            (150, 200, 150): 1,  # Stroma
            (0, 255, 0): 2,      # Blutgefäß
            (200, 0, 0): 3,      # Tumor
            (23, 172, 169): 4,   # Epidermis
            (51, 0, 51): 5       # Nekrose
        }

        color_array = np.array(color_image)
        mask = np.zeros((color_array.shape[0], color_array.shape[1]), dtype=np.uint8)

        for color, class_value in color_to_class_map.items():
            is_color = np.all(color_array == color, axis=-1)
            mask[is_color] = class_value

        return mask

    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        # Fetch image and mask paths from DataFrame
        img_row = self.patch_info.iloc[idx]
        image_path = img_row['path']

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        tissue_mask = None

        # Load tissue mask
        tissue_mask_path = img_row['tissue_mask']
        tissue_mask = Image.open(tissue_mask_path).convert('RGB')  # Load as RGB
        tissue_mask = self.color_image_to_mask(tissue_mask)  # Convert to class mask

        # Apply geometric augmentations to image and tissue mask
        aug_targets = {'tissue_mask': tissue_mask}

        if self.do_geometric_augment:
            transformed_geo = self.geoaug(image=image, **aug_targets)
            image = transformed_geo['image']
            tissue_mask = transformed_geo['tissue_mask']

        # Apply color augmentations to the image
        if self.do_color_augment:
            transformed_col = self.coloraug(image=image)
            image = transformed_col["image"]
            # Apply additional RandAugment
            image = randaugment.distort_image_with_randaugment(image, 3, 5, 'Default')

        # Normalize the image (if normalization is provided)
        if self.normalize:
            image = self.normalize(image=image)['image']

        # Convert image to tensor
        image = torch.from_numpy(image.copy()) / 255 
        tissue_mask = torch.from_numpy(tissue_mask)

        # Convert tissue mask to one-hot encoding using np.eye()
        tissue_mask_one_hot = np.eye(self.num_classes_tissue)[tissue_mask]

        return image.permute(2, 0, 1), tissue_mask_one_hot




