import segmentation_models_pytorch as smp
from cls_model import UNetPlusPlusClassifier
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import yaml
import json
import random
import set_seed
from dataset_nuc import NucleiSegmentationDataset
import utils_nuc as utils
from utils import increment_path

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score
import numpy as np



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# initial_weights_path =
num_classes = 4
class_labels = list(range(0, num_classes))
with open('class_code.json', 'r') as file:
    class_code = json.load(file)

accuracy = smp.metrics.accuracy
iou_score = smp.metrics.iou_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training Segmentation Models.")
    parser.add_argument('--name', type=str, default="cls_nuc",
                        help="Experiment name.")
    parser.add_argument('--specs', type=str, default="./train_specs.yaml")

    opt = parser.parse_args()
    print(opt)

    with open(opt.specs) as f:
        trial_spec = yaml.load(f, Loader=yaml.FullLoader)

    print(opt.name)
    wdir = increment_path(
        Path('./logs/' + opt.name + os.sep))
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # load and store hyper parameters
    with open(opt.specs) as f:
        specs = yaml.load(f, Loader=yaml.FullLoader)
    with open(wdir/'train_specs.yml', 'w') as specs_file:
        yaml.dump(specs, specs_file, default_flow_style=False)
    # weights_path = initial_weights_path


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Doing computations on device = {}'.format(device))

    
    start_epoch = 0
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.00001
    num_classes = 4
    
    
    # Daten-Loader erstellen
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/val.csv')

    train_dataset = NucleiSegmentationDataset(
        train_df,
        add_gray_channel=True,
        use_aug_data = True,
        ttrain = True
    )

    validation_data = NucleiSegmentationDataset(
        val_df,
        add_gray_channel=True, 
    )


    print(f"Anzahl der Daten im Trainingsdatensatz: {len(train_dataset)}")

    total_train_nuclei = sum(len(item) for item in train_dataset)
    print(f"Gesamte Trainingsdaten (extrahierte Nuklei): {total_train_nuclei}")
    
    total_val_nuclei = sum(len(item) for item in validation_data)
    print(f"Gesamte Validierungsdaten (extrahierte Nuklei): {total_val_nuclei}")


    train_loader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=4)

    val_loader = DataLoader(validation_data,
                                       batch_size=32,
                                       sampler=None,
                                       shuffle=False,
                                       num_workers=4)

    
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Validation Loader Batches: {len(val_loader)}")

    ##############################################
    # Modell, Loss und Optimizer instanziieren
    ##############################################

    model = UNetPlusPlusClassifier()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Trainingsschleife: Speichere das Modell nur, wenn der Macro-F1 verbessert wird
    num_epochs = 50
    best_macro_f1 = 0.0
    model_save_path = "best_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        
        # Validierung
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation", leave=False)
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Bestes Modell gespeichert (Macro F1: {macro_f1:.4f})")
        
        scheduler.step()
    
    print("Training abgeschlossen!")

