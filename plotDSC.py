# === plotDSC.py ===

import pandas as pd
import matplotlib.pyplot as plt

file_path = "/EvaluationResults.csv"
df = pd.read_csv(file_path)

print("Column names in the dataset:")
print(df.columns.tolist())

# filtering dice score columns (all columns starting with "dice-")
dice_columns = [col for col in df.columns if col.strip().startswith("dice-")]

print("Filtered dice columns:")
print(dice_columns)

df_dice = df[dice_columns]

null_columns = df_dice.columns[df_dice.isnull().all()]
print(f"Completely null columns (no segmentation data): {null_columns.tolist()}")

# mean Dice score for each organ, ignoring NaN values
mean_dice_scores = df_dice.mean(skipna=True)

# counting the number of valid (non-null) entries for each organ
valid_counts = df_dice.notnull().sum()

print("Number of valid segmentations per organ (after filtering):")
print(valid_counts)

# sorting by Dice scores
mean_dice_scores_sorted = mean_dice_scores.sort_values()

mean_dice_scores_sorted.plot(kind="bar", figsize=(12, 6), title="Average Dice scores per ROI")
plt.ylabel("Average Dice Score")
plt.xlabel("Regions of Interest (ROI)")
plt.tight_layout()
plt.show()
