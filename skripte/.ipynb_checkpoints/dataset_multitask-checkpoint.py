import albumentations as A
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import os
from he_randaugment import randaugment

class PUMAPanopticDataset(data.Dataset):
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
                 normalize=None,
                 geometric_augment: bool = False,
                 color_augment: bool = False,
                 augment_magnitude: int = 2,
                 use_tissue_mask: bool = True,
                 use_nuclei_mask: bool = True,
                 num_classes_tissue=6,
                 num_classes_nuclei=4):

        self.patch_info = patch_info
        self.normalize = normalize
        self.do_color_augment = color_augment
        self.aug_magnitude = augment_magnitude
        self.do_geometric_augment = geometric_augment
        self.use_tissue_mask = use_tissue_mask
        self.use_nuclei_mask = use_nuclei_mask
        self.num_classes_tissue = num_classes_tissue
        self.num_classes_nuclei = num_classes_nuclei

        # Define geometric augmentations
        aug_targets = {}
        if use_tissue_mask:
            aug_targets['tissue_mask'] = 'mask'
        if use_nuclei_mask:
            aug_targets['nuclei_mask'] = 'mask'

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


    def color_image_to_mask_nuclei(self, color_image):
        color_to_class_map = {
            (255, 255, 255): 0,    # Background
            (200, 0, 0): 1,        # nuclei_tumor
            (255, 0, 255): 2,      # TILs (lymphocytes and plasma cells)
            # Merge other cell types into one class (class 3: other_cells)
            (150, 200, 150): 3,    # Other cells (histiocytes, stromal cells, etc.)
            (0, 255, 255): 3,
            (0, 255, 0): 3,
            (89, 165, 113): 3,
            (3, 193, 98): 3,
            (52, 4, 179): 3,
            (51, 0, 51): 3,
            (99, 145, 164): 3
        }

        color_array = np.array(color_image)
        mask = np.zeros((color_array.shape[0], color_array.shape[1]), dtype=np.uint8)

        for color, class_value in color_to_class_map.items():
            is_color = np.all(color_array == color, axis=-1)
            mask[is_color] = class_value

        return mask

    # Function to convert tissue color-coded masks to class-based masks
    def color_image_to_mask(self, color_image):
        color_to_class_map = {
            (255, 255, 255): 0,    # Background
            (150, 200, 150): 1,    # Stroma
            (0, 255, 0): 2,        # Blood vessel
            (200, 0, 0): 3,        # Tumor
            (99, 145, 164): 4,     # Epithelium
            (51, 0, 51): 5         # Necrosis
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

        tissue_mask, nuclei_mask = None, None

        # Load masks based on user input
        if self.use_tissue_mask:
            tissue_mask_path = img_row['tissue_mask']
            tissue_mask = Image.open(tissue_mask_path).convert('RGB')  # Load as RGB
            tissue_mask = self.color_image_to_mask(tissue_mask)  # Convert to class mask

        if self.use_nuclei_mask:
            nuclei_mask_path = img_row['nuclei_mask']
            nuclei_mask = Image.open(nuclei_mask_path).convert('RGB')  # Load as RGB
            nuclei_mask = self.color_image_to_mask_nuclei(nuclei_mask)  # Convert to class mask

        # Apply geometric augmentations to image and both masks (if applicable)
        aug_targets = {}
        if self.use_tissue_mask:
            aug_targets['tissue_mask'] = tissue_mask
        if self.use_nuclei_mask:
            aug_targets['nuclei_mask'] = nuclei_mask

        if self.do_geometric_augment:
            transformed_geo = self.geoaug(image=image, **aug_targets)
            image = transformed_geo['image']
            if self.use_tissue_mask:
                tissue_mask = transformed_geo['tissue_mask']
            if self.use_nuclei_mask:
                nuclei_mask = transformed_geo['nuclei_mask']

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
        return_values = [image.permute(2, 0, 1)]

        # Convert and return tissue mask using np.eye() for one-hot encoding
        if self.use_tissue_mask:
            tissue_mask = torch.from_numpy(tissue_mask)
            tissue_mask_one_hot = np.eye(self.num_classes_tissue)[tissue_mask]  # One-hot encoding
            tissue_mask_one_hot = torch.from_numpy(tissue_mask_one_hot)
            return_values.append(tissue_mask_one_hot)

        # Convert and return nuclei mask using np.eye() for one-hot encoding
        if self.use_nuclei_mask:
            nuclei_mask = torch.from_numpy(nuclei_mask)
            nuclei_mask_one_hot = np.eye(self.num_classes_nuclei)[nuclei_mask]  # One-hot encoding
            nuclei_mask_one_hot = torch.from_numpy(nuclei_mask_one_hot)
            return_values.append(nuclei_mask_one_hot)

        # Return the image and the respective masks
        return tuple(return_values)