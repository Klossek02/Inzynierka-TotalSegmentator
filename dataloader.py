import pandas as pd
import os
import matplotlib.pyplot as plt
import nibabel as nib
import warnings as wr
import numpy as np
import seaborn as sns

wr.filterwarnings('ignore')

file_path = r"C:\Users\magda\Desktop\Studia\INZYNIERKA\Totalsegmentator_dataset_v201\meta.csv"

if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path and try again.")

try:
    df = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1')
except UnicodeDecodeError:
    print("Error reading the CSV file. Check the file encoding or path.")
    raise

df.columns = df.columns.str.strip().str.encode('ascii', 'ignore').str.decode('ascii')

print("Column names:", df.columns)

if 'image_id' not in df.columns:
    print("Column 'image_id' not found in DataFrame. Available columns:", df.columns)
    raise KeyError("The column 'image_id' is missing from the DataFrame.")

print(f"\nFirst 10 rows of the dataframe:")
print(df.head(10))

# Sort by image_id
df = df.sort_values(by=['image_id'])

def display_nifti(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(f"\nFile: {file_path}")
        print("CT scan dimensions:", data.shape)
        print("Data range: min =", data.min(), "max =", data.max())

        mid_sagittal = data.shape[0] // 2
        mid_coronal = data.shape[1] // 2
        mid_axial = data.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Axial slice
        axes[0].imshow(data[:, :, mid_axial], cmap='gray', vmin=data.min(), vmax=data.max())
        axes[0].set_title('Axial')
        axes[0].axis('off')

        # Coronal slice
        axes[1].imshow(data[:, mid_coronal, :], cmap='gray', vmin=data.min(), vmax=data.max())
        axes[1].set_title('Coronal')
        axes[1].axis('off')

        # Sagittal slice
        axes[2].imshow(data[mid_sagittal, :, :], cmap='gray', vmin=data.min(), vmax=data.max())
        axes[2].set_title('Sagittal')
        axes[2].axis('off')

        plt.show()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# central slices for the first 10 images
for index, row in df.head(10).iterrows():
    img_folder = os.path.join(r'C:\Users\magda\Desktop\Studia\INZYNIERKA\Totalsegmentator_dataset_v201', row['image_id'])
    img_file_path = os.path.join(img_folder, 'ct.nii.gz')

    if not os.path.isfile(img_file_path):
        print(f"Image file not found: {img_file_path}")
        continue

    print(f"\n{row['image_id']}: Age: {row['age']}, Gender: {row['gender']}, Pathology: {row['pathology']}, Pathology Location: {row['pathology_location']}")
    display_nifti(img_file_path)

np.random.seed(55)
sample_size = 10
rd_sample = df.sample(n=sample_size)

for index, row in rd_sample.iterrows():
    img_folder = os.path.join(r'C:\Users\magda\Desktop\Studia\INZYNIERKA\Totalsegmentator_dataset_v201', row['image_id'])
    img_file_path = os.path.join(img_folder, 'ct.nii.gz')

    if not os.path.isfile(img_file_path):
        print(f"Image file not found: {img_file_path}")
        continue

    print(f"\n{row['image_id']}: Age: {row['age']}, Gender: {row['gender']}, Pathology: {row['pathology']}, Pathology Location: {row['pathology_location']}")
    display_nifti(img_file_path)

# Exploratory Data Analysis (EDA)

# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['age'], bins=32, kde=True, color='blue')
plt.title('TotalSegmentator Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Gender distribution
gen_counts = df['gender'].value_counts()
colors = ['lightblue', 'lightpink']

plt.figure(figsize=(8, 8))
plt.pie(
    gen_counts,
    labels=["Male", "Female"],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    wedgeprops={"linewidth": 1, "edgecolor": "black"}
)
plt.title('TotalSegmentator Gender Distribution')
plt.show()

# Age distribution by gender
age_bins = [0, 2, 4, 6, 9, 12, 14, 17, 20, 24, 26, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = [
    "1-2", "3-4", "5-6",
    "7-9", "10-12", "13-14",
    "15-17", "18-20", "21-24",
    "25-26", "27-30", "31-40",
    "41-50", "51-60", "61-70",
    "71-80", "81-90", "91-100"
]

df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)

plt.figure(figsize=(12, 6))
sns.histplot(
    data=df,
    x='age_group',
    hue='gender',
    shrink=0.9,
    multiple='dodge',
    palette={'m': 'lightblue', 'f': 'lightpink'},
    stat='count'
)
plt.title('TotalSegmentator Age Distribution by Gender')
plt.xlabel('Age group')
plt.ylabel('Count')
plt.legend(title='Gender:', labels=['Female', 'Male'], loc='upper right')
plt.show()

# Pathology distribution
path_counts = df['pathology'].value_counts()
print("\nPathology distribution:")
print(path_counts)

plt.figure(figsize=(12, 6))
path_counts.plot(kind='bar', color='lightgreen')
plt.title('TotalSegmentator Pathology Distribution')
plt.xlabel('Pathology')
plt.ylabel('Count')
plt.show()

# Split distribution
train_count = len(df[df['split'] == 'train'])
validation_count = len(df[df['split'] == 'val'])
test_count = len(df[df['split'] == 'test'])

print("TotalSegmentator split distribution:")
print(f"Training set: {train_count}")
print(f"Validation set: {validation_count}")
print(f"Testing set: {test_count}")



# import os
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# from monai.transforms import Compose, Lambda
#
# def scale_intensity_range(tensor, a_min, a_max, b_min, b_max, clip=False):
#     """
#     Scale intensity range of a tensor from [a_min, a_max] to [b_min, b_max].
#     """
#     # Avoid division by zero if a_min equals a_max
#     if a_max == a_min:
#         return tensor * (b_max - b_min) + b_min
#     tensor = (tensor - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
#     if clip:
#         tensor = torch.clamp(tensor, b_min, b_max)
#     return tensor
#
# class NiftiDataset(Dataset):
#     def __init__(self, file_path, transforms=None):
#         self.file_path = file_path
#         self.transforms = transforms
#         self.data = self.load_nifti_file(file_path)
#
#     def load_nifti_file(self, file_path):
#         nifti_img = nib.load(file_path)
#         nifti_data = nifti_img.get_fdata()
#         return nifti_data
#
#     def __len__(self):
#         return self.data.shape[2]
#
#     def __getitem__(self, idx):
#         slice_data = self.data[:, :, idx]
#         slice_data = torch.tensor(slice_data, dtype=torch.float32)  # Convert to tensor
#         if self.transforms:
#             slice_data = self.transforms(slice_data)
#         return slice_data
#
# # Define the transformations
# transforms = Compose([
#     Lambda(lambda x: x.unsqueeze(0)),  # Add channel dimension
#     Lambda(lambda x: scale_intensity_range(x, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True))
# ])
#
# file_path = r"C:\Users\magda\Desktop\Studia\INZYNIERKA\Totalsegmentator_dataset_v201\s0003\segmentations\aorta.nii.gz"
#
# # Load the dataset
# dataset = NiftiDataset(file_path, transforms=transforms)
# print(f"Full dataset shape: {dataset.data.shape}")
#
# # Check the range of the original data
# original_min = dataset.data.min()
# original_max = dataset.data.max()
# print(f"Original data range: Min = {original_min}, Max = {original_max}")
#
# # Calculate the indices for the middle third of the dataset
# total_slices = len(dataset)
# start_idx = total_slices // 3
# end_idx = 2 * (total_slices // 3)
# middle_slices = list(range(start_idx, end_idx))
#
# # Filter non-zero slices within the middle third
# non_zero_slices = []
# for i in middle_slices:
#     slice_data = dataset[i]
#     non_zero_count = torch.count_nonzero(slice_data).item()
#     if non_zero_count > 0:
#         non_zero_slices.append(i)
#
# print(f"Non-zero slices in the middle third: {non_zero_slices}")
#
# # Visualize up to 6 non-zero slices from the middle third
# for idx in non_zero_slices[:6]:
#     slice_data = dataset[idx]
#     print(f"Visualizing non-zero slice {idx + 1}")
#     # Print min and max values of slice_data to debug scaling
#     print(f"Slice {idx + 1} - Min value: {slice_data.min().item()}, Max value: {slice_data.max().item()}")
#     plt.imshow(slice_data.squeeze().numpy(), cmap='gray')
#     plt.title(f"Slice {idx + 1}")
#     plt.axis('off')  # Hide axis for clearer visualization
#     plt.show()
