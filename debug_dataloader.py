import torch
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
from dataloader import get_dataloaders

def debug_dataset(dataset, num_samples=5):
    print("[Dataset Debugging]")
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            if sample is None:
                print(f"Sample {i} is None.")
                continue

            image, label = sample['image'], sample['label']
            print(f"[Dataset] Sample {i}:")
            print(f"  Image shape: {image.shape}, Label shape: {label.shape}")
            print(f"  Unique labels in tensor: {torch.unique(label)}")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

def debug_dataloader(dataloader, num_batches=2):
    print("[Dataloader Debugging]")
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        try:
            images, labels = batch['image'], batch['label']
            print(f"[Collate] Batch {i}:")
            print(f"  Batched image shape: {images.shape}, Batched label shape: {labels.shape}")
            print(f"  Unique values in labels: {torch.unique(labels)}")
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

if __name__ == "__main__":
    base_dir = "Totalsegmentator_dataset_v201"
    meta_csv = "Totalsegmentator_dataset_v201/meta.csv"

    train_loader, val_loader, _ = get_dataloaders(base_dir, meta_csv, combine_masks=True, batch_size=4)

    train_ds = train_loader.dataset
    val_ds = val_loader.dataset

    print("=== Debugging train dataset ===")
    debug_dataset(train_ds, num_samples=5)

    print("\n=== Debugging validation dataset ===")
    debug_dataset(val_ds, num_samples=5)

    print("\n=== Debugging train dataloader ===")
    debug_dataloader(train_loader, num_batches=2)

    print("\n=== Debugging validation dataloader ===")
    debug_dataloader(val_loader, num_batches=2)

# WARNING: since the output is big, a good practice is to type in the terminal:
# python debug_dataloader.py > output_debug.txt
