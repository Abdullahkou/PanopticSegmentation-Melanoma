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
from scipy.ndimage import label as nd_label
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import matplotlib.pyplot as plt
 
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
        self.nuclei_data = []

        
        if not os.path.exists(self.augmented_csv_path):
            pass
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


      # Lade alle Nuklei aus den Bildern
        for _, row in tqdm(self.patch_info.iterrows(), total=len(self.patch_info), desc="Processing Patches"):
            img_path = row['path']
            mask_path = row['nuclei_mask']
            
            image = np.array(Image.open(img_path).convert('RGB'))
            tissue_mask = np.load(mask_path) if mask_path.endswith('.npy') else self.color_image_to_mask(Image.open(mask_path).convert('RGB'))
            
            extracted_nuclei = self.extract_nuclei_from_image_and_mask(image, tissue_mask)
            
            for nucleus, label in extracted_nuclei:
                # Normalisierung
                rgb_image = self.normalize(nucleus)

                if self.add_gray_channel:
                    gray_image = cv2.cvtColor(nucleus, cv2.COLOR_RGB2GRAY)
                    gray_image = torch.tensor(gray_image, dtype=torch.float32) / 255.0
                    gray_image = gray_image.unsqueeze(0)  # [1, H, W]
                    nucleus = torch.cat((rgb_image, gray_image), dim=0)
                else:
                    nucleus = rgb_image

                self.nuclei_data.append((nucleus, label))

        print(f"Gesamtzahl extrahierter Nuklei: {len(self.nuclei_data)}")

   
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
            




    def extract_nuclei_from_image_and_mask(self, image, mask, target_size=(64, 64)):
        """
        Extrahiert Nuklei aus einem Bild basierend auf der Ground-Truth-Maske, indem alle Instanzen gelabelt werden.
    
        Parameters:
            image (np.ndarray): Originalbild.
            mask (np.ndarray): Ground-Truth-Maske mit eindeutigen Pixelwerten pro Nukleus.
            target_size (tuple): Zielgröße für die ausgeschnittenen Nuklei (Breite, Höhe).
    
        Returns:
            list: Eine Liste von Tupeln [(nucleus_image, class_label)], wobei nucleus_image das Bild
                  des Nukleus und class_label die Klasse des Nukleus ist.
        """
    
    
        # Sicherstellen, dass die Maske und das Bild die gleichen Dimensionen haben
        if image.shape[:2] != mask.shape:
            raise ValueError("Die Dimensionen von Bild und Maske stimmen nicht überein.")
    
        # Sicherstellen, dass die Maske im richtigen Format vorliegt
    
        # Instanzen in der Maske labeln
        labeled_mask, num_features = nd_label(mask > 0)
    
        # Liste, um Ergebnisse zu speichern
        nuclei_data = []
    
        for instance_id in range(1, num_features + 1):
            # Maske für die aktuelle Instanz
            nucleus_mask = (labeled_mask == instance_id).astype(np.uint8)
    
            # Bounding Box des Nukleus berechnen
            x, y, w, h = cv2.boundingRect(nucleus_mask)
    
            # Nukleus-Bild aus dem Originalbild ausschneiden (mit Sicherheitspuffer)
            buffer = 5 # Optional: Sicherheitsabstand um die Zelle
            x_start = max(x - buffer, 0)
            y_start = max(y - buffer, 0)
            x_end = min(x + w + buffer, image.shape[1])
            y_end = min(y + h + buffer, image.shape[0])
    
            # Ausschneiden des Bildbereichs mit Puffer
            nucleus_image = image[y_start:y_end, x_start:x_end]
    
            # Klasse des Nukleus bestimmen (z. B. anhand des häufigsten Pixelwerts innerhalb der Instanz)
            instance_class = np.argmax(np.bincount(mask[nucleus_mask > 0].flatten()))
    
            # Auf Zielgröße skalieren
            nucleus_resized = cv2.resize(nucleus_image, target_size, interpolation=cv2.INTER_LANCZOS4)
            #nucleus_resized = self.apply_laplacian_sharpening(nucleus_resized)
    
            # Zum Ergebnis hinzufügen
            nuclei_data.append((nucleus_resized, instance_class))
    
        return nuclei_data
        
    def apply_laplacian_sharpening(self, image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = cv2.addWeighted(image, 1.0, cv2.convertScaleAbs(laplacian), -0.5, 0)
        return sharpened
        
     
    def __len__(self):
        return len(self.nuclei_data)

    def __getitem__(self, idx):
        
        return self.nuclei_data[idx]



