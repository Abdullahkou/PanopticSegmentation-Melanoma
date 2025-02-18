import pandas as pd
import albumentations as A
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import os
from he_randaugment import randaugment
import torchvision.transforms as transforms
import cv2
import ast
from tqdm import tqdm
from scipy.ndimage import label

class NucleiSegmentationDataset(data.Dataset):
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
                 add_gray_channel=True,
                 add_gray2=True,
                 use_aug_data = True,
                 ttrain = False,
                 norm = True,
                 normalize=None,
                 geometric_augment: bool = False,
                 color_augment: bool = False,
                 augment_magnitude: int = 1,
                 num_classes_tissue=4,
                 output_dir=r"../augmented_data"): 

        self.patch_info = patch_info
        self.add_gray_channel = add_gray_channel  # Bool für zusätzlichen Graustufen-Kanal
        self.use_aug_data = use_aug_data
        self.ttrain = ttrain
        self.norm = norm
        self.normalize = normalize or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.do_color_augment = color_augment
        self.aug_magnitude = augment_magnitude
        self.do_geometric_augment = geometric_augment
        self.num_classes_tissue = num_classes_tissue
        self.output_dir = output_dir
        self.augmented_csv_path = os.path.join(output_dir, "augmented_patch_info.csv")
        self.add_gray2=add_gray2

        if not os.path.exists(self.augmented_csv_path):
            os.makedirs(self.output_dir, exist_ok=True)
            augmented_patch_info = self.create_augmented_dataset()
        else:
            augmented_patch_info = pd.read_csv(self.augmented_csv_path)

        
        if self.ttrain and self.use_aug_data:
            
            #filtered_augmented_patch_info = self.filter_exclude_value(df=augmented_patch_info, exclude_value=1)
            filtered_augmented_patch_info = augmented_patch_info
            # Concatenate the filtered data
            print(len(filtered_augmented_patch_info))
            self.patch_info = pd.concat([self.patch_info, filtered_augmented_patch_info], ignore_index=True)

        # Define geometric augmentations
        aug_targets = {'nuclei_mask': 'mask'}

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

   
    def filter_exclude_value(self, df, exclude_value=1):
        """
        Filters the DataFrame to exclude rows where the specified column contains the exclude_value.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            exclude_value (int): The value to exclude.
        
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        column_name = "class"
        
        # Ensure the column exists
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        # Filter the DataFrame
        filtered_df = df[df[column_name] != exclude_value]
        
        return filtered_df
        
    def color_image_to_mask(self, color_image):
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
            
    def create_augmented_dataset(self, num_augmentations=2):
        augmented_data = []
    
        for idx in tqdm(range(len(self.patch_info)), desc="Processing aug data"):
            img_row = self.patch_info.iloc[idx]
            image_path = img_row['path']
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
    
            tissue_mask_path = img_row['nuclei_mask']
            tissue_mask = Image.open(tissue_mask_path).convert('RGB')
            tissue_mask = self.color_image_to_mask(tissue_mask)
    
            for i in range(num_augmentations):
                aug_image, aug_mask = image.copy(), tissue_mask.copy()
                aug_targets = {'nuclei_mask': aug_mask}
    
                # Geometrische Augmentationen korrekt einrichten
                geo_transforms = A.Compose([
                        A.RandomRotate90(p=1.0),
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
                        A.Perspective(scale=(0.05, 0.1), p=1.0)
                    ], additional_targets={'nuclei_mask': 'mask'})
                    
                transformed_geo = geo_transforms(image=aug_image, nuclei_mask=aug_mask)
                aug_image = transformed_geo['image']
                aug_mask = transformed_geo['nuclei_mask']

                color_transforms = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.CLAHE(clip_limit=2, p=1.0)
                ], p=1.0)

                transformed_col = color_transforms(image=aug_image)
                aug_image = transformed_col["image"]
    
                aug_image = aug_image.astype(np.uint8)
    
                # Speichern der augmentierten Bilder und Masken
                aug_image_path = os.path.join(self.output_dir, f"aug_image_{img_row['filename']}_{i}.png")
                aug_mask_path = os.path.join(self.output_dir, f"aug_mask_{img_row['filename']}_{i}.npy")
    
                Image.fromarray(aug_image).save(aug_image_path)
                np.save(aug_mask_path, aug_mask)
    
                # Erstellen der Zeile für den augmentierten DataFrame
                aug_row = img_row.copy()
                aug_row['path'] = aug_image_path
                aug_row['nuclei_mask'] = aug_mask_path
                augmented_data.append(aug_row)
    
        augmented_patch_info = pd.DataFrame(augmented_data)
        augmented_patch_info.to_csv(self.augmented_csv_path, index=False)
        return augmented_patch_info


    def compute_hover_maps(self, maske):
        """
        Berechnet HoVer-Maps (horizontale und vertikale Karten) für eine Instanzsegmentierungsmaske.
    
        Args:
            maske (np.ndarray): 2D-Segmentierungsmaske (Höhe x Breite), bei der jede Klasse als Ganzzahl dargestellt wird.
    
        Returns:
            np.ndarray: Ein Array der Form (Höhe, Breite, 2),
                        wobei der erste Kanal die horizontalen und der zweite Kanal die vertikalen Vektoren enthält.
        """
        # Hintergrund ignorieren und Instanzen labeln
        gelabelte_maske, anzahl_instanz = label(maske > 0)  # Jede Instanz erhält eine eindeutige ID
        #print(f"Anzahl der Instanzen: {anzahl_instanz}")
    
        hoehe, breite = maske.shape
        hover_maps = np.zeros((hoehe, breite, 2), dtype=np.float32)
    
        for instanz_id in range(1, anzahl_instanz + 1):  # ID 0 ist Hintergrund
            # Maske für die aktuelle Instanz
            instanz_maske = (gelabelte_maske == instanz_id).astype(np.uint8)
    
            # Schwerpunkt der Instanz berechnen
            coords = np.argwhere(instanz_maske > 0)
            if len(coords) == 0:
                continue
    
            cx = np.mean(coords[:, 1])  # Durchschnitt der x-Koordinaten
            cy = np.mean(coords[:, 0])  # Durchschnitt der y-Koordinaten
    
            # Maximaldistanz für Normierung berechnen
            max_dist = np.max(np.sqrt((coords[:, 1] - cx)**2 + (coords[:, 0] - cy)**2)) + 1e-8
    
            # Berechne horizontale und vertikale Distanzen für jedes Pixel in der Instanz
            for y, x in coords:
                dx = x - cx
                dy = y - cy
                hover_maps[y, x, 0] = dx / max_dist  # Horizontal
                hover_maps[y, x, 1] = dy / max_dist  # Vertikal
    
        return hover_maps

     
    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        # Fetch image and mask paths from DataFrame
        img_row = self.patch_info.iloc[idx]
        image_path = img_row['path']

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        rgb_image = self.normalize(image)  # Normalisiertes RGB-Bild als Tensor

        tissue_mask = None
        tissue_mask_path = img_row['nuclei_mask']


        if tissue_mask_path.endswith('.npy'):
            tissue_mask = np.load(tissue_mask_path)  # Lade die Maske als NumPy-Array
        else:
            tissue_mask = Image.open(tissue_mask_path).convert('RGB')  # Load as RGB
            tissue_mask = self.color_image_to_mask(tissue_mask)  # Convert to class mask

        if self.add_gray_channel and self.add_gray2:

            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_image = torch.tensor(gray_image, dtype=torch.float32) / 255.0
            gray_image = gray_image.unsqueeze(0) 
        
            hoVerMaps = self.compute_hover_maps(tissue_mask)
            hoVerMaps = torch.tensor(hoVerMaps, dtype=torch.float32).permute(2, 0, 1)  # (2, H, B)
            image = torch.cat((rgb_image, hoVerMaps, gray_image), dim=0)  # (5, H, B)

        else: 
            hoVerMaps = self.compute_hover_maps(tissue_mask)
            hoVerMaps = torch.tensor(hoVerMaps, dtype=torch.float32).permute(2, 0, 1)  # (2, H, B)
            image = torch.cat((rgb_image, hoVerMaps), dim=0)  # (5, H, B)

        tissue_mask = torch.from_numpy(tissue_mask)
        tissue_mask_one_hot = np.eye(self.num_classes_tissue)[tissue_mask]
        return image, tissue_mask_one_hot




