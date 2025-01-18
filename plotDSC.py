# === plotDSC.py ===

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\magda\Desktop\EvaluationResults.csv"
df = pd.read_csv(file_path)

# Debug: Ensure column names are correct and check for issues
print("Column names in the dataset:")
print(df.columns.tolist())

# Filter for dice score columns (all columns starting with "dice-")
dice_columns = [col for col in df.columns if col.strip().startswith("dice-")]

# Debug: Print filtered columns
print("Filtered dice columns:")
print(dice_columns)

# Subset the dataframe to only dice score columns
df_dice = df[dice_columns]

# Check for completely null columns
null_columns = df_dice.columns[df_dice.isnull().all()]
print(f"Completely null columns (no segmentation data): {null_columns.tolist()}")

# Calculate the mean Dice score for each organ, ignoring NaN values
mean_dice_scores = df_dice.mean(skipna=True)

# Count the number of valid (non-null) entries for each organ
valid_counts = df_dice.notnull().sum()

# Debugging: Print valid counts for each organ
print("Number of valid segmentations per organ (after filtering):")
print(valid_counts)

# Sort by Dice scores for better visualization
mean_dice_scores_sorted = mean_dice_scores.sort_values()

# Plot the average Dice scores
mean_dice_scores_sorted.plot(kind="bar", figsize=(12, 6), title="Average Dice scores per ROI")
plt.ylabel("Average Dice Score")
plt.xlabel("Regions of Interest (ROI)")
plt.tight_layout()
plt.show()
