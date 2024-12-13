# === Train.py ===

import glob
import os

from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Compose, Resize
from monai.networks.utils import one_hot
from sklearn.metrics import accuracy_score, jaccard_score

from dataloader import get_dataloaders, TotalSeg_Dataset_Tr_Val
from model import get_unet_model

from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import nibabel as nib


def train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=1, use_amp=False, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    num_classes = 118
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(percentile=95, directed=False)

    best_val_loss = float('inf')
    best_metric_epoch = -1
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_ious = []
    val_dices = []
    val_hausdorffs = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        all_dice_scores = []
        all_hausdorff_distances = []
        i = 0
        for batch in train_loader:
            if batch is None:
                print(f"ERROR: skipping batch {i} - invalid samples.")
                i += 1
                continue
            print('batch:', i)
            i += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            softmax = torch.nn.functional.softmax(outputs, dim=1) # converting predictions to probabilities
            labels_one_hot = one_hot(labels, num_classes=num_classes) # one-hot
            dice_score = dice_metric(y_pred=softmax, y=labels_one_hot) # dice score (metrics)
            preds_class = torch.argmax(softmax, dim=1).unsqueeze(1) # converting predictions to one of the classes 
            hausdorff_distance = hausdorff_metric(y_pred=preds_class, y=labels) # hausdorff distance (metrics)
            all_hausdorff_distances.append(hausdorff_distance.item()) 

            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        if all_labels and all_preds:
            all_labels_np = np.concatenate(all_labels).flatten()
            all_preds_np = np.concatenate(all_preds).flatten()
            train_accuracy = accuracy_score(all_labels_np, all_preds_np)
            train_accuracies.append(train_accuracy)
        else:
            train_accuracy = 0

        avg_dice = np.mean(all_dice_scores)
        avg_hausdorff = np.mean(all_hausdorff_distances)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}, Training Accuracy: {train_accuracy}, Training Dice: {avg_dice}, Training Hausdorff: {avg_hausdorff}')

        # validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_preds = []
            all_labels = []
            all_dice_scores = []
            all_hausdorff_distances = []
            val_count = 0
            i = 0
            for batch in val_loader:
                if batch is None:
                    print(f"ERROR: skipping batch {i} - invalid samples.")
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

                        # metrics as above
                        softmax = torch.nn.functional.softmax(outputs, dim=1)
                        labels_one_hot = one_hot(labels, num_classes=num_classes)
                        dice_score = dice_metric(y_pred=softmax, y=labels_one_hot)
                        all_dice_scores.append(dice_score.item())
                        preds_class = torch.argmax(softmax, dim=1).unsqueeze(1)
                        hausdorff_distance = hausdorff_metric(y_pred=preds_class, y=labels)
                        all_hausdorff_distances.append(hausdorff_distance.item())

                        all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                        val_count += 1
                except Exception as e:
                    print(f"Error during validation at epoch {epoch + 1}: {e}")
                    continue

            if val_count > 0:
                val_loss /= val_count
                val_losses.append(val_loss)

                all_labels_np = np.concatenate(all_labels)
                all_preds_np = np.concatenate(all_preds)

                val_accuracy = accuracy_score(all_labels_np.flatten(), all_preds_np.flatten())
                val_accuracies.append(val_accuracy)

                # IoU calculation
                val_iou = jaccard_score(all_labels_np.flatten(), all_preds_np.flatten(), average='macro')
                val_ious.append(val_iou)

                avg_dice = np.mean(all_dice_scores)
                val_dices.append(avg_dice)

                avg_hausdorff = np.mean(all_hausdorff_distances)
                val_hausdorffs.append(avg_hausdorff)

                print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation IoU: {val_iou}, Validation Dice: {avg_dice}, Validation Hausdorff: {avg_hausdorff}')

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

    # plotting training and validation accuracy 
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.show()
    plt.savefig('Training_validation_loss.png')

    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.show()
    plt.savefig('Training_validation_accuracy.png')

    plt.figure()
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='Validation IoU', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU')
    plt.show()
    plt.savefig('validation_IoU.png')


    plt.figure()
    plt.plot(range(1, len(val_dices) + 1), val_dices, label='Validation dice', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Dice coefficient')
    plt.legend()
    plt.title('Validation dice coefficient')
    plt.show()
    plt.savefig('validation_dice.png')

    plt.figure()
    plt.plot(range(1, len(val_hausdorffs) + 1), val_hausdorffs, label='Validation Hausdorff distance', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Hausdorff distance')
    plt.legend()
    plt.title('Validation Hausdorff distance')
    plt.show()
    plt.savefig('validation_hd.png')


def test(model, test_loader, device=None, use_amp=False, save_predictions=False, save_path="test_predictions"): 
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if save_predictions:
        os.makedirs(save_path, exist_ok=True)
        
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                print(f"ERROR: skipping test batch {i} - invalid samples.")
                continue

            inputs = batch["image"].to(device)

            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)

                    if save_predictions:
                        preds = torch.argmax(outputs, dim=1)
                        for idx in range(inputs.size(0)):
                            save_nifti(preds[idx].cpu().numpy(), save_path, index=i * test_loader.batch_size + idx)

            except Exception as e:
                print(f"ERROR: Exception occurred during testing - batch {i}: {e}")
                continue


def save_nifti(volume, path, index=0):
    volume = np.array(volume, dtype=np.float32)
    nifti_image = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti_image, os.path.join(path, f'patient_predicted_{index}.nii.gz'))
    print(f'patient_predicted_{index} is saved', end='\r')

if __name__ == "__main__":
    base_dir = "Totalsegmentator_dataset_v201"
    meta_csv = "Totalsegmentator_dataset_v201/meta.csv"
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, meta_csv)

    print(f"Training dataloader length: {len(train_loader)}")
    print(f"Validation dataloader length: {len(val_loader)}")
    print(f"Test dataloader length: {len(test_loader)}")

    model = get_unet_model(num_classes=118, in_channels=1)
    criterion = DiceLoss(softmax=True, to_onehot_y=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # training, valdation phase
    train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=1, use_amp=True, patience=5)

    # saving trained model
    torch.jit.script(model).save('model.zip')

    # loading best model for testing 
    #best_model = get_unet_model(num_classes = 118, in_channels = 1)
    #best_model.load_state_dict(torch.load("best_metric_model.pth"))
    #best_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # testing phase
    #test(best_model, test_loader, use_amp=True, save_predictions=True, save_path='test_predictions')
