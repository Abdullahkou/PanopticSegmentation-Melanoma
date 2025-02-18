import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import yaml
import json
import set_seed
from model_v2 import UNetPlusPlusMultiTask  # Dein Multitask-Modell
from dataset_multitask import PUMAPanopticDataset  # Dein Multitask-Dataset
from multi_loss import MultiTaskFocalDiceLoss  # Deine Multitask-Verlustfunktion

import utils
from utils import increment_path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Anzahl der Klassen
num_classes_tissue = 6  # Gewebeklassen
num_classes_nuclei = 4  # Zellkernklassen (inklusive Hintergrund)

# Laden der Klassenbezeichnungen
with open('class_code.json', 'r') as file:
    class_code = json.load(file)

accuracy = smp.metrics.accuracy
iou_score = smp.metrics.iou_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training Segmentation Models.")
    parser.add_argument('--name', type=str, default="PSG",
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
    # Speichern der Hyperparameter
    with open(opt.specs) as f:
        specs = yaml.load(f, Loader=yaml.FullLoader)
    with open(wdir/'train_specs.yml', 'w') as specs_file:
        yaml.dump(specs, specs_file, default_flow_style=False)

    start_epoch = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # Laden der Trainings- und Validierungsdaten
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/val.csv')

    set_seed.seed_everything(seed=trial_spec['seed'], workers=False)
    print(trial_spec)

    writer = SummaryWriter(log_dir=os.path.join(wdir, "logs_torch"))
    batch_size = trial_spec['batch_size']

    training_data = PUMAPanopticDataset(
        train_df,
        num_classes_tissue=num_classes_tissue,
        num_classes_nuclei=num_classes_nuclei
    )
    validation_data = PUMAPanopticDataset(
        val_df,
        num_classes_tissue=num_classes_tissue,
        num_classes_nuclei=num_classes_nuclei
    )

    num_samples = int(0.8 * len(training_data))
    len_data = int(num_samples / batch_size)

    weights = torch.DoubleTensor(train_df['weight'].values)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )

    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=trial_spec['num_workers'])

    validation_dataloader = DataLoader(validation_data,
                                       batch_size=16,
                                       sampler=None,
                                       shuffle=False,
                                       num_workers=trial_spec['num_workers'])

    model = UNetPlusPlusMultiTask(num_classes_tissue=num_classes_tissue,
                                  num_classes_nuclei=num_classes_nuclei)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=trial_spec['learning_rate'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200 * len_data / batch_size)

    alpha_tissue = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
    alpha_nuclei = torch.tensor([0.0, 1.0, 3.0, 2.0]).to(device) 
    
    criterion = MultiTaskFocalDiceLoss(
        alpha_tissue=alpha_tissue.to(device),
        alpha_nuclei=alpha_nuclei.to(device),
        gamma=2.0,
        dice_weight=0.5,
        alpha=0.5,
        beta=0.5
        )

    epochs = trial_spec['N_epochs']
    best_loss = np.inf
    best_f1 = 0



    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()
    
        metrics_loss = {"train_loss": [], "val_loss": []}
        metrics = {
            "train_acc_tissue": [], "val_acc_tissue": [],
            "train_iou_tissue": [], "val_iou_tissue": [],
            "train_acc_nuclei": [], "val_acc_nuclei": [],
            "train_iou_nuclei": [], "val_iou_nuclei": []
        }
    
        for (x, y_tissue, y_nuclei) in tqdm(train_dataloader, total=len(train_dataloader), desc="train"):
            x = x.to(device)
            y_tissue = y_tissue.to(device)
            y_nuclei = y_nuclei.to(device)
        
            # Konvertiere One-Hot-encoded Targets zu Klassenindizes
            tissue_target = y_tissue.argmax(dim=3)  # Korrigiert von dim=1 zu dim=3
            nuclei_target = y_nuclei.argmax(dim=3)  # Korrigiert von dim=1 zu dim=3
        
            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward pass
            tissue_output, nuclei_output = model(x)
        
            # Compute predictions
            tissue_pred = torch.argmax(tissue_output, dim=1)  # Shape: [Batch, H, W]
            nuclei_pred = torch.argmax(nuclei_output, dim=1)  # Shape: [Batch, H, W]
        
            # Print tensor shapes for debugging
            #print(f"tissue_output shape: {tissue_output.shape}")  # Erwartet: [Batch, num_classes_tissue, H, W]
            #print(f"tissue_target shape: {tissue_target.shape}")  # Erwartet: [Batch, H, W]
            #print(f"tissue_pred shape: {tissue_pred.shape}")      # Erwartet: [Batch, H, W]
        
            # Berechne den Verlust mit den konvertierten Targets
            loss = criterion(
                tissue_output, nuclei_output, tissue_target, nuclei_target)
    
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                scheduler.step()
    
                metrics_loss["train_loss"].append(loss.item())
    
                # Compute metrics for tissue segmentation
                tp_tissue, fp_tissue, fn_tissue, tn_tissue = smp.metrics.get_stats(
                    tissue_pred, tissue_target, mode='multiclass', num_classes=num_classes_tissue)
                metrics["train_acc_tissue"].append(
                    accuracy(tp_tissue, fp_tissue, fn_tissue, tn_tissue, reduction="micro"))
                metrics["train_iou_tissue"].append(
                    iou_score(tp_tissue, fp_tissue, fn_tissue, tn_tissue, reduction="micro"))
    
                # Compute metrics for nuclei detection
                tp_nuclei, fp_nuclei, fn_nuclei, tn_nuclei = smp.metrics.get_stats(
                    nuclei_pred, nuclei_target, mode='multiclass', num_classes=num_classes_nuclei)
                metrics["train_acc_nuclei"].append(
                    accuracy(tp_nuclei, fp_nuclei, fn_nuclei, tn_nuclei, reduction="micro"))
                metrics["train_iou_nuclei"].append(
                    iou_score(tp_nuclei, fp_nuclei, fn_nuclei, tn_nuclei, reduction="micro"))
    
        # Log training metrics
        train_loss_epoch = np.mean(metrics_loss["train_loss"])
        writer.add_scalar('loss/train', train_loss_epoch, epoch)
    
        # Tissue metrics
        train_acc_tissue_epoch = np.mean(metrics["train_acc_tissue"])
        train_iou_tissue_epoch = np.mean(metrics["train_iou_tissue"])
        writer.add_scalar('accuracy/train_tissue', train_acc_tissue_epoch, epoch)
        writer.add_scalar('iou/train_tissue', train_iou_tissue_epoch, epoch)
    
        # Nuclei metrics
        train_acc_nuclei_epoch = np.mean(metrics["train_acc_nuclei"])
        train_iou_nuclei_epoch = np.mean(metrics["train_iou_nuclei"])
        writer.add_scalar('accuracy/train_nuclei', train_acc_nuclei_epoch, epoch)
        writer.add_scalar('iou/train_nuclei', train_iou_nuclei_epoch, epoch)
        # Validation loop
        model.eval()
        with torch.no_grad():
            metrics_loss["val_loss"] = []
            metrics["val_acc_tissue"] = []
            metrics["val_iou_tissue"] = []
            metrics["val_acc_nuclei"] = []
            metrics["val_iou_nuclei"] = []
        
            for (x, y_tissue, y_nuclei) in tqdm(validation_dataloader, total=len(validation_dataloader), desc="validation"):
                x = x.to(device)
                y_tissue = y_tissue.to(device)
                y_nuclei = y_nuclei.to(device)
        
                # Konvertiere One-Hot-encoded Targets zu Klassenindizes
                tissue_target = y_tissue.argmax(dim=3)  # Korrigiert von dim=1 zu dim=3
                nuclei_target = y_nuclei.argmax(dim=3)  # Korrigiert von dim=1 zu dim=3
        
                # Forward pass
                tissue_output, nuclei_output = model(x)
        
                # Compute predictions
                tissue_pred = torch.argmax(tissue_output, dim=1)  # Shape: [Batch, H, W]
                nuclei_pred = torch.argmax(nuclei_output, dim=1)  # Shape: [Batch, H, W]
        
                # Print tensor shapes for debugging (optional)
                #print(f"tissue_output shape: {tissue_output.shape}")  # Erwartet: [Batch, num_classes_tissue, H, W]
                #print(f"tissue_target shape: {tissue_target.shape}")  # Erwartet: [Batch, H, W]
                #print(f"tissue_pred shape: {tissue_pred.shape}")      # Erwartet: [Batch, H, W]
        
                # Berechne den kombinierten Verlust für die Validierung
                val_loss = criterion(
                    tissue_output, nuclei_output, tissue_target, nuclei_target)
                metrics_loss["val_loss"].append(val_loss.item())
        
                # Compute metrics for tissue segmentation
                tp_tissue, fp_tissue, fn_tissue, tn_tissue = smp.metrics.get_stats(
                    tissue_pred, tissue_target, mode='multiclass', num_classes=num_classes_tissue)
                metrics["val_acc_tissue"].append(
                    accuracy(tp_tissue, fp_tissue, fn_tissue, tn_tissue, reduction="micro"))
                metrics["val_iou_tissue"].append(
                    iou_score(tp_tissue, fp_tissue, fn_tissue, tn_tissue, reduction="micro"))
        
                # Compute metrics for nuclei detection
                tp_nuclei, fp_nuclei, fn_nuclei, tn_nuclei = smp.metrics.get_stats(
                    nuclei_pred, nuclei_target, mode='multiclass', num_classes=num_classes_nuclei)
                metrics["val_acc_nuclei"].append(
                    accuracy(tp_nuclei, fp_nuclei, fn_nuclei, tn_nuclei, reduction="micro"))
                metrics["val_iou_nuclei"].append(
                    iou_score(tp_nuclei, fp_nuclei, fn_nuclei, tn_nuclei, reduction="micro"))
        
        # Log validation metrics
        val_loss_epoch = np.mean(metrics_loss["val_loss"])
        writer.add_scalar('loss/validation', val_loss_epoch, epoch)
        
        # Tissue metrics
        val_acc_tissue_epoch = np.mean(metrics["val_acc_tissue"])
        val_iou_tissue_epoch = np.mean(metrics["val_iou_tissue"])
        writer.add_scalar('accuracy/val_tissue', val_acc_tissue_epoch, epoch)
        writer.add_scalar('iou/val_tissue', val_iou_tissue_epoch, epoch)
        
        # Nuclei metrics
        val_acc_nuclei_epoch = np.mean(metrics["val_acc_nuclei"])
        val_iou_nuclei_epoch = np.mean(metrics["val_iou_nuclei"])
        writer.add_scalar('accuracy/val_nuclei', val_acc_nuclei_epoch, epoch)
        writer.add_scalar('iou/val_nuclei', val_iou_nuclei_epoch, epoch)
        
        # Save model checkpoints
        if epoch % trial_spec['save_model_step'] == 0:
            torch.save(model.state_dict(), wdir / f'checkpoint_{epoch}.pt')
        
        # Speichern des besten Modells basierend auf Validierungsverlust
        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            torch.save(model.state_dict(), wdir / 'best_model.pt')
            print(f'Best model saved with validation loss: {best_loss}')
        
        print(f"Epoch {epoch} completed. Training Loss: {train_loss_epoch}, Validation Loss: {val_loss_epoch}")

    writer.close()