import segmentation_models_pytorch as smp
from MTL_unetpp import MultiTaskUnetPlusPlus
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
import set_seed
from MTL_dataset import MTLDataset
import utils_nuc as utils
from utils import increment_path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# initial_weights_path =

num_classes_nuc = 4
num_classes_tissue = 4


class_labels_nuc = list(range(0, num_classes_nuc))
class_labels_tissue  = list(range(0, num_classes_tissue))

with open('class_code_nuc.json', 'r') as file:
    class_code_nuc = json.load(file)

with open('class_code_tissue.json', 'r') as file:
    class_code_tissue = json.load(file)   

accuracy = smp.metrics.accuracy
iou_score = smp.metrics.iou_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training Segmentation Models.")
    parser.add_argument('--name', type=str, default="MTL",
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


    start_epoch = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Doing computations on device = {}'.format(device))

    # Define Training / Validation data

    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/val.csv')

    set_seed.seed_everything(seed=trial_spec['seed'], workers=False)
    print(trial_spec)

    writer = SummaryWriter(log_dir=os.path.join(wdir, "logs_torch"))
    batch_size = trial_spec['batch_size']

    training_data = MTLDataset(
        train_df,
        add_gray_channel=True
    )

    validation_data = MTLDataset(
        val_df,
        add_gray_channel=True
    )

    num_samples = int(0.7 * len(training_data))

    len_data = int(num_samples / batch_size)
    


    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=trial_spec['num_workers'])

    validation_dataloader = DataLoader(validation_data,
                                       batch_size=16,
                                       sampler=None,
                                       shuffle=False,
                                       num_workers=trial_spec['num_workers'])

    model = MultiTaskUnetPlusPlus( num_classes_task1_tissue= num_classes_tissue,
    num_classes_task2_nuclei = num_classes_nuc,
    in_channels = 4,
    encoder_depth = 4
    )

    model.to(device)

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    optimizer = torch.optim.Adam(g0, lr=trial_spec['learning_rate'])

    # add g1 with weight_decay
    optimizer.add_param_group(
        {'params': g1, 'weight_decay': trial_spec['weight_decay']})
    optimizer.add_param_group({'params': g2})  # add g2 (biases)

    del g0, g1, g2

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200 * len_data / batch_size)

    epochs = trial_spec['N_epochs']
    best_loss = np.inf

    nuc_best_f1 = 0
    nuc_class_f1 = {}  # keep track of individual class f1 score
    nuc_best_class_f1 = {}

    tissue_best_f1 = 0
    tissue_class_f1 = {}  # keep track of individual class f1 score
    tissue_best_class_f1 = {}

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()

        # Unfreeze model
        # if epoch >= trial_spec['unfreeze_from_epoch']:
        # model.encoder.requires_grad_(True)

        # Training loop
        nuc_metrics_loss = {"train_loss": [], "val_loss": [],
                        'train_loss_supervised': [],
                        'train_loss_contrastive': []}
        nuc_metrics = {"train_acc": [], "val_acc": [],
                   "train_iou": [], "val_iou": [],
                   "train_auc": [], "val_auc": []}

        nuc_train_pos_neg = torch.zeros((4, num_classes_nuc))
        nuc_val_pos_neg = torch.zeros((4, num_classes_nuc))



        tissue_metrics_loss = {"train_loss": [], "val_loss": [],
                        'train_loss_supervised': [],
                        'train_loss_contrastive': []}
        tissue_metrics = {"train_acc": [], "val_acc": [],
                   "train_iou": [], "val_iou": [],
                   "train_auc": [], "val_auc": []}

        tissue_train_pos_neg = torch.zeros((4, num_classes_tissue))
        tissue_val_pos_neg = torch.zeros((4, num_classes_tissue))



        for (x, y_tissue, y_nuclei)in tqdm(train_dataloader,
                           total=len(train_dataloader), desc="train"):

            x = x.to(device)

            y_tissue = y_tissue.to(device)
            y_nuclei = y_nuclei.to(device)
        
            y_tissue = y_tissue.long()
            y_nuclei = y_nuclei.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            tissue_output, nuclei_output = model(x)

            tissue_output.to(device)
            nuclei_output.to(device)

            loss1 = utils.loss(tissue_output.permute(0, 2, 3, 1), y_tissue)
            loss2 = utils.loss(nuclei_output.permute(0, 2, 3, 1), y_nuclei, t_type="nuc")
            loss = loss1 + loss2 

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                scheduler.step()

                nuc_metrics_loss["train_loss"].append(loss.detach())
                tissue_metrics_loss["train_loss"].append(loss.detach())


                tissue_output = torch.sigmoid(tissue_output)
                tissue_pred = torch.argmax(tissue_output, dim=1)
                
                nuclei_output = torch.sigmoid(nuclei_output)
                nuclei_pred = torch.argmax(nuclei_output, dim=1)
                
                y_tissue = torch.argmax(y_tissue, dim=3)
                y_nuclei = torch.argmax(y_nuclei, dim=3)


                tp, fp, fn, tn = smp.metrics.get_stats(
                    tissue_pred, y_tissue, mode='multiclass', num_classes=num_classes_tissue)
                tissue_train_pos_neg += torch.stack(
                    [torch.sum(tp, 0), torch.sum(fp, 0), torch.sum(fn, 0), torch.sum(tn, 0)])
                tissue_metrics["train_acc"].append(
                    accuracy(tp, fp, fn, tn, reduction="micro"))
                tissue_metrics["train_iou"].append(
                    iou_score(tp, fp, fn, tn, reduction="micro"))
                


                tp, fp, fn, tn = smp.metrics.get_stats(
                    nuclei_pred, y_nuclei, mode='multiclass', num_classes=num_classes_nuc)
                nuc_train_pos_neg += torch.stack(
                    [torch.sum(tp, 0), torch.sum(fp, 0), torch.sum(fn, 0), torch.sum(tn, 0)])
                nuc_metrics["train_acc"].append(
                    accuracy(tp, fp, fn, tn, reduction="micro"))
                nuc_metrics["train_iou"].append(
                    iou_score(tp, fp, fn, tn, reduction="micro"))



        loss_epoch_train = torch.mean(torch.stack(nuc_metrics_loss["train_loss"]))
        print(
            f'Loss on epoch {epoch}: {loss_epoch_train}')

        writer.add_scalar('loss/train', loss_epoch_train, epoch)

        nuc_accuracy_epoch_train = torch.mean(torch.stack(nuc_metrics["train_acc"]))

        tissue_accuracy_epoch_train = torch.mean(torch.stack(tissue_metrics["train_acc"]))

        scores = utils.f1_score(
            nuc_train_pos_neg[0], nuc_train_pos_neg[1], nuc_train_pos_neg[2])
        f1_epoch_train = torch.mean(scores[1:])
        iou_epoch_train = torch.mean(torch.stack(nuc_metrics["train_iou"]))
        writer.add_scalar('accuracy/train', nuc_accuracy_epoch_train, epoch)
        writer.add_scalar('iou/train', iou_epoch_train, epoch)
        writer.add_scalar('f1/train', f1_epoch_train, epoch)
        for class_idx in class_labels_nuc:
            writer.add_scalar(
                f'class_train/{list(class_code_nuc)[class_idx]}_f1_train', scores[class_idx], epoch)

        writer.add_scalar('learning_rate/train',
                          optimizer.__dict__.get('param_groups')[0].get('lr'),
                          epoch)
        

        scores = utils.f1_score(
            tissue_train_pos_neg[0], tissue_train_pos_neg[1], tissue_train_pos_neg[2])
        f1_epoch_train = torch.mean(scores[1:])
        iou_epoch_train = torch.mean(torch.stack(tissue_metrics["train_iou"]))
        writer.add_scalar('accuracy/train1', tissue_accuracy_epoch_train, epoch)
        writer.add_scalar('iou/train1', iou_epoch_train, epoch)
        writer.add_scalar('f1/train1', f1_epoch_train, epoch)
        for class_idx in class_labels_tissue:
            writer.add_scalar(
                f'class_train/{list(class_code_tissue)[class_idx]}_f1_train', scores[class_idx], epoch)

        writer.add_scalar('learning_rate/train1',
                          optimizer.__dict__.get('param_groups')[0].get('lr'),
                          epoch)






        # Validation loop
        losses_on_epoch_val = []

        model = model.eval()
        nuc_confusion_matrix = np.zeros(
            (num_classes_nuc, num_classes_nuc), dtype=np.float64)
        

        tissue_confusion_matrix = np.zeros(
            (num_classes_tissue, num_classes_tissue), dtype=np.float64)

        with torch.no_grad():
            for (x, y_tissue, y_nuclei) in tqdm(validation_dataloader,
                               total=len(
                                   validation_dataloader),
                               desc="validation"):
                x = x.to(device)

                y_tissue = y_tissue.to(device)
                y_nuclei = y_nuclei.to(device)
            
                y_tissue = y_tissue.long()
                y_nuclei = y_nuclei.long()

                tissue_output, nuclei_output = model(x)

                tissue_output.to(device)
                nuclei_output.to(device)

                val_loss1 = utils.loss(tissue_output.permute(0, 2, 3, 1), y_tissue)
                val_loss2 = utils.loss(nuclei_output.permute(0, 2, 3, 1), y_nuclei, t_type="nuc")

                current_val_loss = (val_loss1 + val_loss2)/2



                if not torch.isnan(current_val_loss.detach()):
                    nuc_metrics_loss["val_loss"].append(current_val_loss)

                    tissue_output = torch.sigmoid(tissue_output)
                    tissue_pred = torch.argmax(tissue_output, dim=1)
                    
                    nuclei_output = torch.sigmoid(nuclei_output)
                    nuclei_pred = torch.argmax(nuclei_output, dim=1)
                    
                    y_tissue = torch.argmax(y_tissue, dim=3)
                    y_nuclei = torch.argmax(y_nuclei, dim=3)

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        tissue_pred, y_tissue, mode='multiclass', num_classes=num_classes_tissue)
                    tissue_val_pos_neg += torch.stack([torch.sum(tp, 0), torch.sum(
                        fp, 0), torch.sum(fn, 0), torch.sum(tn, 0)])
                    tissue_metrics["val_acc"].append(
                        accuracy(tp, fp, fn, tn, reduction="micro"))
                    tissue_metrics["val_iou"].append(
                        iou_score(tp, fp, fn, tn, reduction="micro"))
                    

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        nuclei_pred, y_nuclei, mode='multiclass', num_classes=num_classes_nuc)
                    nuc_val_pos_neg += torch.stack([torch.sum(tp, 0), torch.sum(
                        fp, 0), torch.sum(fn, 0), torch.sum(tn, 0)])
                    nuc_metrics["val_acc"].append(
                        accuracy(tp, fp, fn, tn, reduction="micro"))
                    nuc_metrics["val_iou"].append(
                        iou_score(tp, fp, fn, tn, reduction="micro"))



                    if epoch % trial_spec['plot_cm_step'] == 0:
                        pass

            validation_loss = torch.mean(torch.stack(nuc_metrics_loss["val_loss"]))
            print(f'validation loss on epoch {epoch}: {validation_loss}')
            writer.add_scalar('loss/validation', validation_loss, epoch)

            accuracy_epoch_val = torch.mean(torch.stack(nuc_metrics["val_acc"]))
            scores = utils.f1_score(
                nuc_val_pos_neg[0], nuc_val_pos_neg[1], nuc_val_pos_neg[2])
            f1_epoch_val = torch.mean(scores[1:])
            iou_epoch_val = torch.mean(torch.stack(nuc_metrics["val_iou"]))
            writer.add_scalar('accuracy/val', accuracy_epoch_val, epoch)
            writer.add_scalar('iou/val', iou_epoch_val, epoch)
            writer.add_scalar('f1/val', f1_epoch_val, epoch)

            for class_idx in class_labels_nuc:
                writer.add_scalar(
                    f'class_val/{list(class_code_nuc)[class_idx]}_f1_val', scores[class_idx], epoch)
                nuc_class_f1[class_idx] = scores[class_idx]

            f1_task1_nuc = f1_epoch_val
            iou_task1_nuc = iou_epoch_val

            accuracy_epoch_val = torch.mean(torch.stack(tissue_metrics["val_acc"]))
            scores = utils.f1_score(
                tissue_val_pos_neg[0], tissue_val_pos_neg[1], tissue_val_pos_neg[2])
            f1_epoch_val = torch.mean(scores[1:])
            iou_epoch_val = torch.mean(torch.stack(tissue_metrics["val_iou"]))
            writer.add_scalar('accuracy/val1', accuracy_epoch_val, epoch)
            writer.add_scalar('iou/val1', iou_epoch_val, epoch)
            writer.add_scalar('f1/val1', f1_epoch_val, epoch)

            for class_idx in class_labels_tissue:
                writer.add_scalar(
                    f'class_val/{list(class_code_tissue)[class_idx]}_f1_val', scores[class_idx], epoch)
                tissue_class_f1[class_idx] = scores[class_idx]
        

            f1_epoch_val = (f1_task1_nuc + f1_epoch_val) / 2 
            iou_epoch_val = (iou_task1_nuc + iou_epoch_val )/2

        best_f1_file = wdir / f'best_f1.pt'
        best_iou_file = wdir / f'best_iou.pt'

        if epoch == start_epoch:
            best_f1_mean = 0
            best_iou_mean = 0

        if best_f1_mean < f1_epoch_val:
            print(f'F1 score improved from {best_f1_mean} to {f1_epoch_val}')
            best_f1_mean = f1_epoch_val
            torch.save(model.state_dict(), best_f1_file)

        if best_iou_mean < iou_epoch_val:
            print(
                f'Accuracy improved from {best_iou_mean} to {iou_epoch_val}')
            best_iou_mean = iou_epoch_val
            torch.save(model.state_dict(), best_iou_file)

        for class_idx in class_labels_nuc:
            #early_stopper = early_stopper_list[class_idx]
            #stop_criterion = early_stopper(model, class_f1[class_idx])
            best = wdir / f'best_{class_idx}.pt'

            if epoch == start_epoch:
                best_f1 = 0
                nuc_best_class_f1[class_idx] = 0
            else:
                best_f1 = nuc_best_class_f1[class_idx]

            if best_f1 < nuc_class_f1[class_idx]:
                improvement = "Yes"
                values_before = f'{best_f1:.4f}'
                values_after = f'{nuc_class_f1[class_idx]:.4f}'
                nuc_best_class_f1[class_idx] = nuc_class_f1[class_idx]
                torch.save(model.state_dict(), best)
            else:
                improvement = "No"
                values_before = f'{best_f1:.4f}'
                values_after = f'{nuc_class_f1[class_idx]:.4f}'

            print(f"| Class {class_idx:2d} | Improvement: {improvement} | "
                  f"Values Before: {values_before} | Values After: {values_after} | "
                  )

        for class_idx in class_labels_tissue:
            #early_stopper = early_stopper_list[class_idx]
            #stop_criterion = early_stopper(model, class_f1[class_idx])
            best = wdir / f'best_{class_idx}.pt'

            if epoch == start_epoch:
                best_f1 = 0
                tissue_best_class_f1[class_idx] = 0
            else:
                best_f1 = tissue_best_class_f1[class_idx]

            if best_f1 < tissue_class_f1[class_idx]:
                improvement = "Yes"
                values_before = f'{best_f1:.4f}'
                values_after = f'{tissue_class_f1[class_idx]:.4f}'
                tissue_best_class_f1[class_idx] = tissue_class_f1[class_idx]
                torch.save(model.state_dict(), best)
            else:
                improvement = "No"
                values_before = f'{best_f1:.4f}'
                values_after = f'{tissue_class_f1[class_idx]:.4f}'

            print(f"| Class {class_idx:2d} | Improvement: {improvement} | "
                  f"Values Before: {values_before} | Values After: {values_after} | "
                  )