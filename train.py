# === Train.py ===

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, jaccard_score


from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

from dataloader import get_dataloaders, TotalSeg_Dataset_Tr_Val
from model import get_unet_model


def train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=50, use_amp=False, patience=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    num_classes = 118  # no. of classes + background 

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
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        batch_idx = 0

        # training phase
        for batch in train_loader:
            if batch is None:
                batch_idx += 1
                continue
            batch_idx += 1

            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                # Labels: [B,1,D,H,W], class idxs
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # metrics forecast
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int8)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(np.int8))

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # calculatning training accuracy without the background 
        if all_labels and all_preds:
            all_labels_np = np.concatenate(all_labels).flatten()
            all_preds_np = np.concatenate(all_preds).flatten()
            mask = all_labels_np != 0
            filtered_labels = all_labels_np[mask]
            filtered_preds = all_preds_np[mask]
            train_accuracy = accuracy_score(filtered_labels, filtered_preds)
            train_accuracies.append(train_accuracy)
        else:
            train_accuracy = 0

        print(f'Epoch {epoch + 1}/{num_epochs}, Training loss: {epoch_loss:.4f}, Training accuracy: {train_accuracy:.4f}')

        # validation phase
        model.eval()
        with torch.no_grad():
          val_loss = 0
          all_preds = []
          all_labels = []

          dice_metric.reset()

          for batch in val_loader:
            if batch is None:
              continue

            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              val_loss += loss.item()

              preds = torch.argmax(outputs, dim=1).to(device)  # [B, D, H, W]
              all_preds.append(preds.cpu().numpy().astype(np.int8))
              all_labels.append(labels.cpu().numpy().astype(np.int8))

              # one-hot of predictions and labels 
              preds_one_hot = one_hot(preds.unsqueeze(1), num_classes=num_classes)
              labels_one_hot = one_hot(labels, num_classes=num_classes)

              # update metrics but do not get values yet 
              dice_metric(y_pred=preds_one_hot, y=labels_one_hot)

        if len(val_loader) > 0:
          val_loss /= len(val_loader)
          val_losses.append(val_loss)

          # aggregating dice_metric results through all validation batches 
          dice_score = dice_metric.aggregate()  # now, we have reduced result (scalar)
          dice_metric.reset()  # cleaning the metric before the next epoch arises

          # calculating remaining metrics 
          all_labels_np = np.concatenate(all_labels)
          all_preds_np = np.concatenate(all_preds)
          mask = all_labels_np.flatten() != 0
          filtered_labels = all_labels_np.flatten()[mask]
          filtered_preds = all_preds_np.flatten()[mask]

          val_accuracy = accuracy_score(filtered_labels, filtered_preds)
          val_accuracies.append(val_accuracy)

          val_iou = jaccard_score(filtered_labels, filtered_preds, average='macro')
          val_ious.append(val_iou)

          # now, dice_score is scalar/ 1-element tensor 
          avg_dice = dice_score.item()
          val_dices.append(avg_dice)

          print(f'Epoch {epoch + 1}/{num_epochs}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}, Validation IoU: {val_iou:.4f}, Validation DSC: {avg_dice:.4f}')

          if val_loss < best_val_loss:
              best_val_loss = val_loss
              best_metric_epoch = epoch + 1
              epochs_no_improve = 0
              torch.save(model.state_dict(), "best_metric_model.pth")
              print("Saved new best metric model.")
          else:
            epochs_no_improve += 1
        else:
            print("No validation data available.")

        scheduler.step(val_loss)

        # early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_metric_epoch}")

    # plots 
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('training_validation_accuracy.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='Validation IoU', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU')
    plt.savefig('validation_IoU.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, len(val_dices) + 1), val_dices, label='Validation DSC', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()
    plt.title('Validation DSC')
    plt.savefig('validation_dice.png')
    plt.show()



def test(model, test_loader, device=None, use_amp=False, save_predictions=False, save_path="test_predictions"):

    num_classes = 118

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = DiceLoss(softmax=True, to_onehot_y=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    test_loss = 0
    all_preds = []
    all_dice_scores = []
    test_count = 0

    if save_predictions:
        os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                print(f"Skipping test batch {i} - invalid samples.")
                continue

            inputs = batch["image"].to(device)

            try:

                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(inputs)

                    if save_predictions:
                        preds = torch.argmax(outputs, dim=1)
                        for idx in range(inputs.size(0)):
                            save_nifti(preds[idx], save_path, index=i * test_loader.batch_size + idx)


            except Exception as e:
                print(f"ERROR: Exception occurred during testing - batch {i}: {e}")
                continue



def save_nifti(volume, path, index=0):
    volume = np.array(volume, dtype=np.int16)
    nifti_image = nib.Nifti1Image(volume, np.eye(4))
    filename = os.path.join(path, f'patient_predicted_{index}.nii.gz')
    nib.save(nifti_image, filename)
    print(f'patient_predicted_{index}.nii.gz is saved.')


if __name__ == "__main__":
    base_dir = "C:/Users/Dell/Downloads/Totalsegmentator_dataset_v201"  # here, the relative path used
    meta_csv = "C:/Users/Dell/Downloads/Totalsegmentator_dataset_v201/meta.csv"

    # data loaders for training, validation, testing phase
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, meta_csv, combine_masks=True, batch_size=1, num_workers=2)

    model = get_unet_model(num_classes=118, in_channels=1)

    criterion = DiceLoss(softmax=True, to_onehot_y=True)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # training 
    train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=50, use_amp=True, patience=10)

    # saving the best model
    torch.jit.script(model).save('model.zip')

    # loading the best model for testing 
    #best_model = get_unet_model(num_classes=118, in_channels=1)
    #best_model.load_state_dict(torch.load("best_metric_model.pth"))
    #best_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #print("Best model loaded for testing.")

    # testing 
    #test(best_model, test_loader, use_amp=True, save_predictions=True, save_path='test_predictions')

