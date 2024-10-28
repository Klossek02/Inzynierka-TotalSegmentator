# === Train.py ===

import glob
import os

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Resize
from sklearn.metrics import accuracy_score, jaccard_score

from dataloader import get_dataloaders, TotalSeg_Dataset_Tr_Val
from model import get_unet_model

from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import nibabel as nib

def train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=15, use_amp=False, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    best_val_loss = float('inf')
    best_metric_epoch = -1
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_ious = []
    val_dices = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        i = 0
        for batch in train_loader:
            if batch is None:
                print(f"Skipping batch {i} due to invalid samples.")
                i += 1
                continue
            print('batch:', i)
            i += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                print('inputs size:', inputs.size())
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        if all_labels and all_preds:
            all_labels = np.concatenate(all_labels).flatten()
            all_preds = np.concatenate(all_preds).flatten()
            train_accuracy = accuracy_score(all_labels, all_preds)
            train_accuracies.append(train_accuracy)
        else:
            train_accuracy = 0

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}, Training Accuracy: {train_accuracy}')


        # validation phase 
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_preds = []
            all_labels = []
            val_count = 0
            i = 0
            for batch in val_loader:
                if batch is None:
                    print(f"Skipping batch {i} due to invalid samples.")
                    i += 1
                    continue
                print(f"Validation batch: {i}")
                i += 1
                inputs, labels = batch["image"].to(device), batch["label"].to(device)

                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                        val_count += 1
                except Exception as e:
                    print(f"Error during validation at epoch {epoch + 1}: {e}")
                    continue


            if val_count > 0:
                val_loss /= val_count
                val_losses.append(val_loss)

                all_labels = np.concatenate(all_labels)
                all_preds = np.concatenate(all_preds)

                val_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
                val_accuracies.append(val_accuracy)

                # calculating IoU and DSC
                val_iou = jaccard_score(all_labels.flatten(), all_preds.flatten(), average='macro')
                val_ious.append(val_iou)

                # converting to torch tensors with the correct shape
                all_labels_tensor = torch.tensor(all_labels, dtype=torch.int64).unsqueeze(1).to(device)
                all_preds_tensor = torch.tensor(all_preds, dtype=torch.int64).unsqueeze(1).to(device)

                # computing Dice score for each batch element
                dice_scores = []
                for i in range(all_preds_tensor.shape[0]):
                    dice_score = dice_metric(y_pred=all_preds_tensor[i:i + 1], y=all_labels_tensor[i:i + 1]).item()
                    dice_scores.append(dice_score)
                val_dice = np.mean(dice_scores)
                val_dices.append(val_dice)

                print(
                    f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation IoU: {val_iou}, Validation DSC: {val_dice}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metric_epoch = epoch + 1
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), "best_metric_model.pth")
                    print("Saved new best metric model")
                else:
                    epochs_no_improve += 1
            else:
                print("No validation data available")

        scheduler.step(val_loss)

        # early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best validation loss: {best_val_loss} at epoch {best_metric_epoch}")

    # plotting training and validation loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # plotting training and validation accuracy
    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # plotting validation IoU
    plt.figure()
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='Validation IoU', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU')
    plt.show()

    # plotting validation dSC
    plt.figure()
    plt.plot(range(1, len(val_dices) + 1), val_dices, label='Validation DSC', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()
    plt.title('Validation DSC')
    plt.show()

def save_nifti(volume, path, index=0):
    volume = np.array(volume.detach().cpu()[0], dtype=np.float32)
    volume = nib.Nifti1Image(volume, np.eye(4))
    nib.save(volume, os.path.join(path, f'patient_predicted_{index}.nii.gz'))
    print(f'patient_predicted_{index} is saved', end='\r')

if __name__ == "__main__":
    base_dir = "Totalsegmentator_dataset_v201"
    meta_csv = "Totalsegmentator_dataset_v201/meta.csv"
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, meta_csv)

    print(f"Training data loader length: {len(train_loader)}")
    print(f"Validation data loader length: {len(val_loader)}")

    model = get_unet_model(num_classes=1, in_channels=1)
    criterion = DiceLoss(sigmoid=True, to_onehot_y=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # training, validation
    train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=15, use_amp=True, patience=3)

    torch.jit.script(model).save('model.zip')
