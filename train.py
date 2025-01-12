# === train.py ===

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import torch

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from sklearn.metrics import accuracy_score, jaccard_score
from torch import optim
from torch.utils.data import DataLoader

from dataloader import get_dataloaders, TotalSeg_Dataset_Tr_Val
from model import get_unet_model


def train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=8, use_amp=False, patience=5):
    """
    Method to train a model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # in case GPU is available, else use CPU
    model = model.to(device)  # source: https://discuss.pytorch.org/t/understanding-model-to-device/123662
    scaler = torch.amp.GradScaler(enabled=use_amp) # initializing automatic mixed precision; source: https://pytorch.org/docs/stable/amp.html
    num_classes = 118 # number of classes defined in TotalSegmentator_v201 dataset

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)  # initializing dice metric; source: https://docs.monai.io/en/stable/metrics.html

    best_val_loss = float('inf') # initializing best validation loss
    best_metric_epoch = -1 # initializing best metric epoch
    epochs_no_improve = 0 # initializing epochs with no improvement

    train_losses = [] # initializing list to store training losses 
    val_losses = [] # -=- validation losses
    train_accuracies = [] # -=- training accuracies
    val_accuracies = [] # -=- validation accuracies
    val_ious = [] # -=- validation IoUs
    val_dices = [] # -=- validation Dices

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}.")
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        batch_idx = 0

        # training loop:
        for batch in train_loader:
            if batch is None:
                batch_idx += 1
                continue
            batch_idx += 1

            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad() # zeroing gradients; source: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int8)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().astype(np.int8))

        epoch_loss /= len(train_loader) 
        train_losses.append(epoch_loss)

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

        # validation phase:
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

                    preds = torch.argmax(outputs, dim=1).to(device)  # [batch_size, depth, height, width]; source: https://pytorch.org/docs/main/generated/torch.argmax.html
                    all_preds.append(preds.cpu().numpy().astype(np.int8))
                    all_labels.append(labels.cpu().numpy().astype(np.int8))

                    preds_one_hot = one_hot(preds.unsqueeze(1), num_classes=num_classes) # source: https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
                    labels_one_hot = one_hot(labels, num_classes=num_classes)

                    dice_metric(y_pred=preds_one_hot, y=labels_one_hot)

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            dice_score = dice_metric.aggregate()
            dice_metric.reset()

            all_labels_np = np.concatenate(all_labels) # source: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
            all_preds_np = np.concatenate(all_preds)
            mask = all_labels_np.flatten() != 0 # source: https://numpy.org/doc/stable/reference/generated/numpy.flatten.html
            filtered_labels = all_labels_np.flatten()[mask]
            filtered_preds = all_preds_np.flatten()[mask]

            val_accuracy = accuracy_score(filtered_labels, filtered_preds)
            val_accuracies.append(val_accuracy) # source: https://numpy.org/doc/stable/reference/generated/numpy.append.html

            val_iou = jaccard_score(filtered_labels, filtered_preds, average='macro')
            val_ious.append(val_iou)

            avg_dice = dice_score.item()
            val_dices.append(avg_dice)

            print(f'Epoch {epoch + 1}/{num_epochs}, Validation loss: {val_loss:.4f}, '
                  f'Validation accuracy: {val_accuracy:.4f}, Validation IoU: {val_iou:.4f}, '
                  f'Validation DSC: {avg_dice:.4f}')

            if val_loss < best_val_loss: # if the current validation loss is less than the best validation loss
                best_val_loss = val_loss
                best_metric_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_metric_model.pth")
                print("New best metric model has been saved.")
            else:
                epochs_no_improve += 1
        else:
            print("No validation data is available.")

        scheduler.step(val_loss)

        # early stopping: 
        if epochs_no_improve >= patience: # if there is no improvement in the validation accuracy (applied to avoid overfitting)
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_metric_epoch}.")

    # plotting training, validation loss:  # source: https://matplotlib.org/stable/api/matplotlib_configuration_api.html --> all matplotlib functions used below
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.savefig('training_validation_loss.png')
    plt.show()

    # plotting training, validation accuracy:
    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.savefig('training_validation_accuracy.png')
    plt.show()

    # plotting validation IoU:
    plt.figure()
    plt.plot(range(1, len(val_ious) + 1), val_ious, label='Validation IoU', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU')
    plt.savefig('validation_IoU.png')
    plt.show()

    # plotting validation dice score:
    plt.figure()
    plt.plot(range(1, len(val_dices) + 1), val_dices, label='Validation DSC', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()
    plt.title('Validation DSC')
    plt.savefig('validation_dice.png')
    plt.show()


def test(model, test_loader, device=None, use_amp=False, save_predictions=False, save_path="test_predictions"):
    """
    Method to test a model - generating predictions.
    """

    # The logic would be similar to the one in training and validation function, except that there are no labels present 

    num_classes = 118

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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
                    preds = torch.argmax(outputs, dim=1)  # [batch_size, depth, height, width]; source: https://pytorch.org/docs/main/generated/torch.argmax.html

                    if save_predictions:
                        for idx in range(inputs.size(0)):
                            save_nifti(preds[idx], save_path, index=i * test_loader.batch_size + idx)

            except Exception as e:
                print(f"ERROR: Exception occurred during testing - batch {i}: {e}")
                continue

    print("\n=== TESTING PHASE DONE ===")
    print(f"Predicted segmentations are saved in the '{save_path}' directory.")


def save_nifti(volume, path, index=0):
    """
    Method for saving a segmentation volume as a NIfTI file.
    """
    volume = np.array(volume, dtype=np.int16)  # source: https://numpy.org/doc/2.1/reference/generated/numpy.array.html
    nifti_image = nib.Nifti1Image(volume, np.eye(4)) # source: https://nipy.org/nibabel/reference/nibabel.nifti1.html
    filename = os.path.join(path, f'patient_predicted_{index}.nii.gz') # source: https://docs.python.org/3/library/os.path.html
    nib.save(nifti_image, filename) # source: https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
    print(f'patient_predicted_{index}.nii.gz is saved.')


def visualize_predictions(save_path, num_samples=5):  # no. of samples can be adjusted according to the preferences 
    """
    Method to visualize predicted NIfTI files.
  
    """
    predicted_files = sorted([f for f in os.listdir(save_path) if f.endswith('.nii.gz')])

    for i, file in enumerate(predicted_files[:num_samples]):
        filepath = os.path.join(save_path, file)
        nifti_img = nib.load(filepath)
        data = nifti_img.get_fdata()

        # we select a middle slice for visualization:
        slice_idx = data.shape[2] // 2
        slice_img = data[:, :, slice_idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(slice_img, cmap='gray')
        plt.title(f'Prediction Slice for {file}')
        plt.axis('off')
        plt.show()


def visualize_side_by_side(test_loader, model, save_path, device='cpu', num_samples=5): # no. of samples can be adjusted according to the preferences 
    """
    Method for visualizing input images and their corresponding predictions (side by side).
    """
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            inputs = batch["image"].to(device)

            try:
                with torch.amp.autocast(device_type='cuda', enabled=False):  
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
                    input_image = inputs.cpu().numpy()[0, 0]

                    # We selecting a middle slice once again:
                    slice_idx = input_image.shape[1] // 2
                    input_slice = input_image[:, slice_idx, :]
                    pred_slice = preds[:, slice_idx, :]

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(input_slice, cmap='gray')
                    axes[0].set_title('CT image')
                    axes[0].axis('off')

                    axes[1].imshow(pred_slice, cmap='jet')
                    axes[1].set_title('Predicted segmentation')
                    axes[1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f'prediction_{i}.png'))
                    plt.show()

            except Exception as e:
                print(f"ERROR: Exception occurred during visualization - batch {i}: {e}.")
                continue


if __name__ == "__main__":

    # directories and parameters used:
    base_dir = "Totalsegmentator_dataset_v201"
    meta_csv = "Totalsegmentator_dataset_v201/meta.csv"
    batch_size = 1
    num_workers = 1
    num_epochs = 8
    use_amp = True
    patience = 5

    # dataloader part:
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, meta_csv, combine_masks=True, batch_size=batch_size, num_workers=num_workers
)

    # model initialization:
    model = get_unet_model(num_classes=118, in_channels=1)

    # loss, optimizer, scheduler: 
    criterion = DiceLoss(softmax=True, to_onehot_y=True)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4) # implementing Adam algorithm; source: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5, verbose=True
    ) # reducing learning rate when metrics do not improve; source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html


    # to star training process, let us uncomment the below code:
    # train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=num_epochs, use_amp=use_amp, patience=patience)


    # loading best model for testing; source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    best_model = get_unet_model(num_classes = 118, in_channels = 1)
    best_model.load_state_dict(torch.load("best_metric_model.pth"))
    best_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    # testing process, saving predictions (if we don't want to test the model, we can comment the below line and uncomment training one):
    test(best_model, test_loader, use_amp=use_amp, save_predictions=True, save_path='test_predictions')


    # visualizing predictions (some of them, here 5 random predictions):
    visualize_predictions(save_path='test_predictions', num_samples=5)

    # visualizing side-by-side comparisons (only if input images are accessible):
    visualize_side_by_side(test_loader=test_loader, model=best_model, save_path='test_visualizations', device=torch.device('cpu'), num_samples=5)
