# === evaluate.py ===

# modified file from the TotalSegmentator repository: https://github.com/wasserth/TotalSegmentator/blob/master/resources/evaluate.py
import sys
from pathlib import Path
from functools import partial
import pandas as pd
from p_tqdm import p_map
import nibabel as nib
import numpy as np
from totalsegmentator.map_to_binary import class_map_5_parts
from metrics import compute_surface_distances, compute_surface_dice_at_tolerance



def dice_score(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    return (2 * intersect) / (denominator + 1e-6)


def calc_metrics(subject, gt_dir, pred_dir, class_map):
    metrics = {"subject": subject}
    for idx, roi_name in class_map.items():
        # building file paths for ground truth and prediction
        gt_file = gt_dir / subject / "segmentations" / f"{roi_name}.nii.gz"
        pred_file = pred_dir / subject / f"{roi_name}.nii.gz"

        # checking whether both files exist
        if not gt_file.exists() or not pred_file.exists():
            print(f"Missing file for organ '{roi_name}' in subject '{subject}'")
            metrics[f"dice-{roi_name}"] = np.NaN
            metrics[f"surface_dice_3-{roi_name}"] = np.NaN
            continue

        # loading ground truth and prediction files
        try:
            gt = nib.load(gt_file).get_fdata()
            pred = nib.load(pred_file).get_fdata()
        except Exception as e:
            print(f"Error loading data for organ '{roi_name}' in subject '{subject}': {e}")
            metrics[f"dice-{roi_name}"] = np.NaN
            metrics[f"surface_dice_3-{roi_name}"] = np.NaN
            continue

        # conversion to boolean
        gt = gt.astype(bool)
        pred = pred.astype(bool)

        # checking for empty/ invalid data
        if np.sum(gt) == 0:
            print(f"Ground truth for organ '{roi_name}' in subject '{subject}' is empty.")
        if np.sum(pred) == 0:
            print(f"Prediction for organ '{roi_name}' in subject '{subject}' is empty.")
        if gt.max() == 0:
            print(f"No valid segmentation in ground truth for organ '{roi_name}' in subject '{subject}'.")

        # compute metrics (mentioned in the thesis)
        if gt.max() > 0:
            metrics[f"dice-{roi_name}"] = dice_score(gt, pred)
            sd = compute_surface_distances(gt, pred, [1.5, 1.5, 1.5])
            metrics[f"surface_dice_3-{roi_name}"] = compute_surface_dice_at_tolerance(sd, 3.0)
        else:
            metrics[f"dice-{roi_name}"] = np.NaN
            metrics[f"surface_dice_3-{roi_name}"] = np.NaN

    return metrics


if __name__ == "__main__":
    gt_dir = "/Totalsegmentator_dataset_v201"
    pred_dir = "/Predictions" # output predictions dir
    class_map = class_map_5_parts["class_map_part_organs"]

    print("Class map contents:")
    print(class_map)


    subjects = [s.name for s in gt_dir.iterdir() if s.is_dir()][:1135] # number of predictions segmented
    print(f"Processing {len(subjects)} subjects...")


    results = p_map(partial(calc_metrics, gt_dir=gt_dir, pred_dir=pred_dir, class_map=class_map), subjects, num_cpus=8)

    df = pd.DataFrame(results)

    output_file = "/EvaluationResults.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # checking for null values per organ
    null_counts = df.isnull().sum()
    print("Null values per organ:")
    print(null_counts)

    # checking for valid segmentations per organ
    valid_counts = df.notnull().sum()
    print("Valid segmentations per organ:")
    print(valid_counts)
