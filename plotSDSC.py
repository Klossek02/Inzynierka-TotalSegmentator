# === plotSDSC.py ===

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_surface_dice(file_path):
    """
    Parses the evaluation output file to extract surface dice scores for organs.
    Args:
        file_path (str): Path to the evaluation output file.
    Returns:
        dict: A dictionary with organ names as keys and a list of surface dice scores as values.
    """
    # Read the file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract columns for surface dice metrics
    surface_dice_columns = [col for col in df.columns if col.startswith('surface_dice_3-')]
    surface_dice_data = {col.replace('surface_dice_3-', ''): df[col].dropna().tolist() for col in surface_dice_columns}

    return surface_dice_data


def plot_surface_dice(surface_dice_data):
    """
    Plots the surface dice scores for each organ.
    Args:
        surface_dice_data (dict): Dictionary with organ names as keys and surface dice scores as values.
    """
    plt.figure(figsize=(12, 6))

    # Calculate mean surface dice for each organ
    mean_surface_dice = {organ: np.mean(scores) for organ, scores in surface_dice_data.items()}
    sorted_organs = sorted(mean_surface_dice, key=mean_surface_dice.get, reverse=True)

    # Organ names and their respective mean surface dice scores
    organs = sorted_organs
    dice_values = [mean_surface_dice[organ] for organ in organs]

    # Plot
    plt.barh(organs, dice_values, color='lightcoral')
    plt.xlabel('Mean Surface Dice Score', fontsize=12)
    plt.ylabel('Organs', fontsize=12)
    plt.title('Mean Surface Dice scores per organ', fontsize=16)
    plt.gca().invert_yaxis()  # Invert the y-axis to show the highest scores at the top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# File path to the evaluation output
file_path = r"C:\Users\magda\Desktop\EvaluationResults.csv"

# Parse and plot surface dice scores
surface_dice_data = parse_surface_dice(file_path)
plot_surface_dice(surface_dice_data)
