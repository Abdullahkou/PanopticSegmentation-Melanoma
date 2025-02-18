import segmentation_models_pytorch as smp
from model_nuc import HalfDualDecUNetPlusPlus
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# initial_weights_path =

num_classes = 4
input_ch = 4
class_labels = list(range(0, num_classes))
with open('class_code.json', 'r') as file:
    class_code = json.load(file)

accuracy = smp.metrics.accuracy
iou_score = smp.metrics.iou_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training Segmentation Models.")
    parser.add_argument('--name', type=str, default="Nuclei_seg",
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

    training_data = NucleiSegmentationDataset(
        train_df,
        add_gray_channel=True,
        use_aug_data = True,
        ttrain = True
    )

    validation_data = NucleiSegmentationDataset(
        val_df,
        add_gray_channel=True, 
        ttrain = False
    )

    random_idx = random.choice(range(len(training_data)))
    image, tissue_mask_one_hot = training_data[random_idx]
    
    # Ausgabe der Formen
    print("Image Shape:", image.shape)           # Erwartete Ausgabe: (4, H, W) falls der Graustufenkanal hinzugefügt wurde
    print("Mask Shape:", tissue_mask_one_hot.shape)  #

    num_samples = int(0.8 * len(training_data))

    len_data = int(num_samples / batch_size)
    
    #weights = torch.DoubleTensor(train_df['weight'].values)
    # weights[weights == 938.033331424317] = 150


    #sampler = WeightedRandomSampler(
     #    weights=weights,
     #    num_samples=num_samples,
     #    replacement=True
     #)



    #sampler = RandomSampler(train_df, replacement=True, num_samples=num_samples)

    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=trial_spec['num_workers'])

    validation_dataloader = DataLoader(validation_data,
                                       batch_size=16,
                                       sampler=None,
                                       shuffle=False,
                                       num_workers=trial_spec['num_workers'])

    model = HalfDualDecUNetPlusPlus(in_channel=input_ch)
    # state_dict = torch.load(weights_path)
    # model.load_state_dict(state_dict, strict=False)
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
    #optimizer = torch.optim.RMSprop(g0, lr=trial_spec['learning_rate'])

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
    best_f1 = 0
    class_f1 = {}  # keep track of individual class f1 score
    best_class_f1 = {}

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()

        # Unfreeze model
        # if epoch >= trial_spec['unfreeze_from_epoch']:
        # model.encoder.requires_grad_(True)

        # Training loop
        metrics_loss = {"train_loss": [], "val_loss": [],
                        'train_loss_supervised': [],
                        'train_loss_contrastive': []}
        metrics = {"train_acc": [], "val_acc": [],
                   "train_iou": [], "val_iou": [],
                   "train_auc": [], "val_auc": []}

        train_pos_neg = torch.zeros((4, num_classes))
        val_pos_neg = torch.zeros((4, num_classes))

        for (x, y) in tqdm(train_dataloader,
                           total=len(train_dataloader), desc="train"):

            x = x.to(device)
            y = y.to(device)
            y = y.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(x)
            output.to(device)

            loss = utils.loss(output.permute(0, 2, 3, 1), y)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
                scheduler.step()

                metrics_loss["train_loss"].append(loss.detach())

                output_pb = torch.sigmoid(output)
                output = torch.argmax(output_pb, dim=1)
                y = torch.argmax(y, dim=3)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    output, y, mode='multiclass', num_classes=num_classes)
                train_pos_neg += torch.stack(
                    [torch.sum(tp, 0), torch.sum(fp, 0), torch.sum(fn, 0), torch.sum(tn, 0)])
                metrics["train_acc"].append(
                    accuracy(tp, fp, fn, tn, reduction="micro"))
                metrics["train_iou"].append(
                    iou_score(tp, fp, fn, tn, reduction="micro"))

        loss_epoch_train = torch.mean(torch.stack(metrics_loss["train_loss"]))
        print(
            f'Loss on epoch {epoch}: {loss_epoch_train}')

        writer.add_scalar('loss/train', loss_epoch_train, epoch)

        accuracy_epoch_train = torch.mean(torch.stack(metrics["train_acc"]))
        scores = utils.f1_score(
            train_pos_neg[0], train_pos_neg[1], train_pos_neg[2])
        f1_epoch_train = torch.mean(scores[1:])
        iou_epoch_train = torch.mean(torch.stack(metrics["train_iou"]))
        writer.add_scalar('accuracy/train', accuracy_epoch_train, epoch)
        writer.add_scalar('iou/train', iou_epoch_train, epoch)
        writer.add_scalar('f1/train', f1_epoch_train, epoch)
        for class_idx in class_labels:
            writer.add_scalar(
                f'class_train/{list(class_code)[class_idx]}_f1_train', scores[class_idx], epoch)

        writer.add_scalar('learning_rate/train',
                          optimizer.__dict__.get('param_groups')[0].get('lr'),
                          epoch)

        # Validation loop
        losses_on_epoch_val = []

        model = model.eval()
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.float64)
        with torch.no_grad():
            for (x, y) in tqdm(validation_dataloader,
                               total=len(
                                   validation_dataloader),
                               desc="validation"):
                x = x.to(device)
                y = y.to(device)
                y = y.long()

                output = model(x)
                output.to(device)

                current_val_loss = utils.loss(output.permute(0, 2, 3, 1), y)
                # current_val_loss = utils.loss(
                #     output, y)

                if not torch.isnan(current_val_loss.detach()):
                    metrics_loss["val_loss"].append(current_val_loss)

                    output_pb = torch.sigmoid(output)
                    output = torch.argmax(output_pb, dim=1)
                    y = torch.argmax(y, dim=3)

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        output, y, mode='multiclass', num_classes=num_classes)
                    val_pos_neg += torch.stack([torch.sum(tp, 0), torch.sum(
                        fp, 0), torch.sum(fn, 0), torch.sum(tn, 0)])
                    metrics["val_acc"].append(
                        accuracy(tp, fp, fn, tn, reduction="micro"))
                    metrics["val_iou"].append(
                        iou_score(tp, fp, fn, tn, reduction="micro"))

                    if epoch % trial_spec['plot_cm_step'] == 0:
                        y_cm = y
                        true_labels = y_cm.view(-1).long()
                        pred_labels = output.view(-1).long()
                        indices = num_classes * true_labels + pred_labels
                        m = torch.bincount(indices, minlength=num_classes **
                                           2).type(torch.float32)

                        # Ausgabe der eindeutigen Werte in true_labels und pred_labels
                        #print("Eindeutige Werte in true_labels:", torch.unique(true_labels))
                        #print("Eindeutige Werte in pred_labels:", torch.unique(pred_labels))

                        m = m.reshape(num_classes, num_classes)
                        confusion_matrix += m.cpu().numpy()

            validation_loss = torch.mean(torch.stack(metrics_loss["val_loss"]))
            print(f'validation loss on epoch {epoch}: {validation_loss}')
            writer.add_scalar('loss/validation', validation_loss, epoch)
            accuracy_epoch_val = torch.mean(torch.stack(metrics["val_acc"]))
            scores = utils.f1_score(
                val_pos_neg[0], val_pos_neg[1], val_pos_neg[2])
            f1_epoch_val = torch.mean(scores[1:])
            iou_epoch_val = torch.mean(torch.stack(metrics["val_iou"]))
            writer.add_scalar('accuracy/val', accuracy_epoch_val, epoch)
            writer.add_scalar('iou/val', iou_epoch_val, epoch)
            writer.add_scalar('f1/val', f1_epoch_val, epoch)

            if epoch % trial_spec['plot_cm_step'] == 0:
                sum_over_rows = confusion_matrix.sum(axis=1, keepdims=True)
                normalized_confusion_matrix = confusion_matrix / sum_over_rows
                cm = utils.plot_cm(normalized_confusion_matrix)
                writer.add_figure("Confusion matrix", cm, epoch)
                # save mask
                utils.plot_and_save_mask_comparison(
                    x, y_cm, output, utils.class_map, epoch, opt.name)
            for class_idx in class_labels:
                writer.add_scalar(
                    f'class_val/{list(class_code)[class_idx]}_f1_val', scores[class_idx], epoch)
                class_f1[class_idx] = scores[class_idx]

        # save after every x epochs starting when contrastive loss is used
        if epoch % trial_spec['save_model_step'] == 0:
            torch.save(model.state_dict(), wdir / f'checkpoint_{epoch}.pt')

        best_f1_file = wdir / f'best_f1.pt'
        best_acc_file = wdir / f'best_acc.pt'

        if epoch == start_epoch:
            best_f1_mean = 0
            best_acc_mean = 0

        if best_f1_mean < f1_epoch_val:
            print(f'F1 score improved from {best_f1_mean} to {f1_epoch_val}')
            best_f1_mean = f1_epoch_val
            torch.save(model.state_dict(), best_f1_file)

        if best_acc_mean < accuracy_epoch_val:
            print(
                f'Accuracy improved from {best_acc_mean} to {accuracy_epoch_val}')
            best_acc_mean = accuracy_epoch_val
            torch.save(model.state_dict(), best_acc_file)

        for class_idx in class_labels:
            #early_stopper = early_stopper_list[class_idx]
            #stop_criterion = early_stopper(model, class_f1[class_idx])
            best = wdir / f'best_{class_idx}.pt'

            if epoch == start_epoch:
                best_f1 = 0
                best_class_f1[class_idx] = 0
            else:
                best_f1 = best_class_f1[class_idx]

            if best_f1 < class_f1[class_idx]:
                improvement = "Yes"
                values_before = f'{best_f1:.4f}'
                values_after = f'{class_f1[class_idx]:.4f}'
                best_class_f1[class_idx] = class_f1[class_idx]
                torch.save(model.state_dict(), best)
            else:
                improvement = "No"
                values_before = f'{best_f1:.4f}'
                values_after = f'{class_f1[class_idx]:.4f}'

            print(f"| Class {class_idx:2d} | Improvement: {improvement} | "
                  f"Values Before: {values_before} | Values After: {values_after} | "
                  )
