# === plot_IoU.py ===

import csv
import matplotlib.pyplot as plt
import numpy as np


def parse_and_calculate_iou(file_path):
    """
    Parses the evaluation CSV file and calculates IoU scores from Dice coefficients.
    Args:
        file_path (str): Path to the evaluation output file.
    Returns:
        dict: A dictionary with organ names as keys and a list of IoU scores as values.
    """
    organ_iou = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # header row

        # mapping organ names to their indices in the header for Dice cols
        organ_indices = {header: idx for idx, header in enumerate(headers) if header.startswith('dice-')}

        for row in reader:
            for organ, idx in organ_indices.items():
                try:
                    dice_score = float(row[idx])
                    # IoU calculation (from the formula)
                    iou = dice_score / (2 - dice_score)
                    if organ not in organ_iou:
                        organ_iou[organ] = []
                    organ_iou[organ].append(iou)
                except (ValueError, IndexError):
                    continue
    return organ_iou


def plot_organ_iou(organ_iou):
    """
    Plots the IoU scores for each organ.
    Args:
        organ_iou (dict): Dictionary with organ names as keys and IoU scores as values.
    """
    plt.figure(figsize=(12, 6))

    # mean IoU for each organ
    mean_iou = {organ: np.mean(scores) for organ, scores in organ_iou.items()}
    sorted_organs = sorted(mean_iou, key=mean_iou.get, reverse=True)

    # Organ names, respective IoU scores
    organs = [organ.replace('dice-', '').replace('_', ' ').capitalize() for organ in sorted_organs]
    iou_values = [mean_iou[organ] for organ in sorted_organs]

    plt.barh(organs, iou_values, color='lightgreen')
    plt.xlabel('Mean IoU Score', fontsize=12)
    plt.ylabel('Organs', fontsize=12)
    plt.title('Mean IoU scores per organ', fontsize=16)
    plt.gca().invert_yaxis()  # show highest IoU at the top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

file_path = "/EvaluationResults.csv"

iou_data = parse_and_calculate_iou(file_path)

plot_organ_iou(iou_data)
