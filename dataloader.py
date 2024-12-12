# === Dataloader.py ===
# Below the module's details on how MONAI dataloader works:
# https://docs.monai.io/en/0.5.2/_modules/monai/data/dataloader.html

import pandas as pd 
import nibabel as nib
import numpy as np
import os 
import subprocess
import glob 
import torch 
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityd,
    RandSpatialCropd, RandRotate90d, RandAffined,
    RandGaussianNoised, ResizeWithPadOrCropd
)
from monai.data import Dataset, CacheDataset

# we are working on the classification of various body parts, grouped into 117 anatomical structures. 
# our aim is to simplify things by combining masks for these structures into single binary mask (eg. lung lobes into whole lung)
# the exact grouping is based on the README here: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TotalSegmentator_v2.md

# note that for some classes the names differ from the standarized anatomical names. The mapping can be found here: https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py

# classes of TotalSegmentator v2 dataset:
total = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right",
        25: "sacrum",
        26: "vertebrae_S1",
        27: "vertebrae_L5",
        28: "vertebrae_L4",
        29: "vertebrae_L3",
        30: "vertebrae_L2",
        31: "vertebrae_L1",
        32: "vertebrae_T12",
        33: "vertebrae_T11",
        34: "vertebrae_T10",
        35: "vertebrae_T9",
        36: "vertebrae_T8",
        37: "vertebrae_T7",
        38: "vertebrae_T6",
        39: "vertebrae_T5",
        40: "vertebrae_T4",
        41: "vertebrae_T3",
        42: "vertebrae_T2",
        43: "vertebrae_T1",
        44: "vertebrae_C7",
        45: "vertebrae_C6",
        46: "vertebrae_C5",
        47: "vertebrae_C4",
        48: "vertebrae_C3",
        49: "vertebrae_C2",
        50: "vertebrae_C1",
        51: "heart",
        52: "aorta",
        53: "pulmonary_vein",
        54: "brachiocephalic_trunk",
        55: "subclavian_artery_right",
        56: "subclavian_artery_left",
        57: "common_carotid_artery_right",
        58: "common_carotid_artery_left",
        59: "brachiocephalic_vein_left",
        60: "brachiocephalic_vein_right",
        61: "atrial_appendage_left",
        62: "superior_vena_cava",
        63: "inferior_vena_cava",
        64: "portal_vein_and_splenic_vein",
        65: "iliac_artery_left",
        66: "iliac_artery_right",
        67: "iliac_vena_left",
        68: "iliac_vena_right",
        69: "humerus_left",
        70: "humerus_right",
        71: "scapula_left",
        72: "scapula_right",
        73: "clavicula_left",
        74: "clavicula_right",
        75: "femur_left",
        76: "femur_right",
        77: "hip_left",
        78: "hip_right",
        79: "spinal_cord",
        80: "gluteus_maximus_left",
        81: "gluteus_maximus_right",
        82: "gluteus_medius_left",
        83: "gluteus_medius_right",
        84: "gluteus_minimus_left",
        85: "gluteus_minimus_right",
        86: "autochthon_left",
        87: "autochthon_right",
        88: "iliopsoas_left",
        89: "iliopsoas_right",
        90: "brain",
        91: "skull",
        92: "rib_left_1",
        93: "rib_left_2",
        94: "rib_left_3",
        95: "rib_left_4",
        96: "rib_left_5",
        97: "rib_left_6",
        98: "rib_left_7",
        99: "rib_left_8",
        100: "rib_left_9",
        101: "rib_left_10",
        102: "rib_left_11",
        103: "rib_left_12",
        104: "rib_right_1",
        105: "rib_right_2",
        106: "rib_right_3",
        107: "rib_right_4",
        108: "rib_right_5",
        109: "rib_right_6",
        110: "rib_right_7",
        111: "rib_right_8",
        112: "rib_right_9",
        113: "rib_right_10",
        114: "rib_right_11",
        115: "rib_right_12",
        116: "sternum",
        117: "costal_cartilages"
    }

# class for loading training, validation images and corresponding segmentations
# we decided to keep the testing set separate since the data processing in this phase differ from one in training and validation steps due to lack of labels

class TotalSeg_Dataset_Tr_Val(Dataset):

    def __init__(self, img_paths, lbl_paths, cmb_masks = False, transform = None): # initialization the dataset with image and label paths and any transforms 
        assert len(img_paths) == len(lbl_paths), 'ERROR: Mismatched length of images and labels' # labels and images must have the same length
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.cmb_masks = cmb_masks
        self.transform = transform
        
    def __len__(self): # returns total number of images 
        return len(self.img_paths)

    def __getitem__(self, idx): # loading image and label paths for given index 
        img_path = self.img_paths[idx]  # get img file path for current index 
        lbl_path = self.lbl_paths[idx]  # -=- for lbl 

        try:
            # since labels have a default path defined as: "Totalsegmentator_dataset_v201\s****\segmentations",
            # we need to ensure that we can access the *.nii.gz files from that directory (s****) as these will be necessary for model training and ct scan visualization,
            # not the folder 'segmentations' itself. 
        
            lbl_files = glob.glob(os.path.join(lbl_path, '*.nii.gz')) 

            # checking if the label file exists 
            if len(lbl_files) == 0:
                raise FileNotFoundError(f'ERROR: missing label for {img_path}.')

            if self.cmb_masks: # case when need to combine mask into a single label 
                cmb_lbl_path = self.combine_masks(lbl_path)
                lbl_files = cmb_lbl_path # making sure that lbl_files contains only combined mask

            # LoadImaged transform can accept file paths instead of data so I don't load the data beforehand
            data = { 'image': img_path, 'label': lbl_files } # organizing data into a dictionary (in case we want to add more data later)

            if self.transform is not None: # if transform is provided
                print(f"Before transform: image path: {img_path}, label path: {lbl_files}")
                data = self.transform(data) # applying any transforms
                print(f"After transform: image shape: {data['image'].shape}, label shape: {data['label'].shape}")
            
            return data

        except Exception as e:
            print(f'ERROR: incorrect processing index {idx}: {e}.')
        return None


    # this function combines individual labels into a single multiple-class mask
    def combine_masks(self, lbl_dir): 
        out_path = os.path.join(lbl_dir, 'combined_mask.nii.gz')  # https://docs.python.org/3/library/os.path.html

        if not os.path.exists(out_path):  # check if the combined mask already exists
            mask_files = glob.glob(os.path.join(lbl_dir, '*.nii.gz')) # get all nii.gz files (mask files)
            combined_mask = None

            # creating one multiple-class mask: 
            for class_id, organ_name in total.items():
                struct_path = os.path.join(lbl_dir, f"{organ_name}.nii.gz")
                if os.path.exists(struct_path):    
                    mask = nib.load(struct_path).get_fdata() # https://nipy.org/nibabel/images_and_memory.html - load mask data            
                    if combined_mask is None:
                        combined_mask = np.zeros_like(mask) # https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html - generating empty mask
                    combined_mask[mask > 0] = class_id # any non-zero value in the current mask will be assigned to the class label
            
            if combined_mask is not None:
                affine = nib.load(mask_files[0]).affine # loading affine matrix: https://medium.com/@junfeng142857/affine-transformation-why-3d-matrix-for-a-2d-transformation-8922b08bce75
                combined_mask_img = nib.Nifti1Image(combined_mask.astype(np.uint8), affine)
                nib.save(combined_mask_img, out_path)

            else:
                combined_mask = np.zeros((128,128,128), dtype=np.uint8) # https://numpy.org/doc/2.1/reference/generated/numpy.zeros.html    
                nifti = nib.Nifti1Image(combined_mask, np.eye(4))
                nib.save(nifti, out_path)
            
        return out_path # returning path to combined mask file 
      

# class for loading test images 
class TotalSeg_Dataset_Ts(Dataset):

    def __init__(self,img_paths, transform = None): # initialization the dataset with image paths and any transforms 
        self.img_paths = img_paths 
        self.transform = transform 

    def __len__(self): # returns tot. number of images 
        return len(self.img_paths)

    def __getitem__(self, idx): # loading image paths for given index 
        img_path = self.img_paths[idx]   # get img file path for current index 
        data = {'image': img_path } # organizing data into a dictionary (in case we want to add more data later)

        if self.transform is not None: # if transform is provided 
            data = self.transform(data) # applying any transform 

        return data


# in this step we create a collate function which handles "None" batches. 
    # this custom function returns either dict ('image': image_tensor of size shape [1, 277, 277, 95], 'label': label_tensor of size shape [117, 277, 277, 95]) or None in case 
    # all samples are invalid 
    # https://lukesalamone.github.io/posts/custom-pytorch-collate/
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/_utils/collate.html#default_collate
def batch_collate_fn_tr_val(batch):  # here we called arg 'batch' instead of data

    # first, let us filter out invalid samples (samples that are None or have 'image' or 'label' as None)
    filtered_batch = [sample for sample in batch if sample is not None]

    if len(filtered_batch) == 0: # if not valid samples present 
        return None 

    # then, let us group the samples by 'image' and 'label'
    images = [sample['image'] for sample in filtered_batch]
    labels =  [sample['label'] for sample in filtered_batch]

    # finally, let us stack the 'image' and 'label' tensors
    batched_images = torch.stack(images, dim=0)  # shape: [batch_size, 1, 277, 277, 95]
    batched_labels = torch.stack(labels, dim=0)  # shape: [batch_size, 117, 277, 277, 95]
    print(f"batched_images.shape: {batched_images.shape}, batched_labels.shape: {batched_labels.shape}")

    return {'image': batched_images, 'label': batched_labels}


def batch_collate_fn_test(batch):  # here we called arg 'batch' instead of data
    # first, let us filter out invalid samples (samples that are None or have 'image' or 'label' as None)
    filtered_batch = [sample for sample in batch if sample is not None]

    if len(filtered_batch) == 0: # if not valid samples present
        return None

    # then, let us group the samples by 'image' and 'label'
    images = [sample['image'] for sample in filtered_batch]

    # finally, let us stack the 'image' and 'label' tensors
    batched_images = torch.stack(images, dim=0)  # shape: [batch_size, 1, 277, 277, 95]
    print(f"batched_images.shape: {batched_images.shape}")

    return {'image': batched_images}


# function that loads and preapres data for training, validation and testing 
"""
    base_dir: base directory where img data is stored 
    meta_csv: path to csv file containing meta information for train/test/val splits
    cmb_masks: if True, combines individual masks into a single binary mask. If False, otherwise
    batch_size: no. of samples per batch 
    num_workers: no. of workers for loading data 
 """

def get_dataloaders(base_dir, meta_csv, combine_masks = True, batch_size = 1, num_workers = 2):
        # helper function to get paths for imgs and their corresponding lbl directories 
        def get_img_lbl_paths(ids):
            img = [] # list of img paths 
            lbl = [] # list of lbl paths 
            for img_id in ids:
                img_path = os.path.join(base_dir, img_id, 'ct.nii.gz')
                lbl_dir = os.path.join(base_dir, img_id, 'segmentations')

                if os.path.exists(img_path) and os.path.exists(lbl_dir):
                    img.append(img_path)
                    lbl.append(lbl_dir)
                else:
                    print(f'ERROR: missing files for {img_id}. Image path: {img_path}, label path: {lbl_dir}')
            return img, lbl

        # load meta.csv file to determine train/test/val split
        try:
            meta_df = pd.read_csv(meta_csv, delimiter = ';')
            print(f'SUCCESS. Metadata has been loaded from {meta_csv}.')
        except Exception as e:
            print(f'ERROR. Metadata failed to load from {meta_csv}.')
            return None, None, None 
        

        # let us show the first 10 rows to see whether the metadata is loaded correctly
        print('Metadata preview:')
        print(meta_df.head(10))

        # now after confirming, extract train/test/val IDs 
        train_ids = meta_df[meta_df['split'] == 'train']['image_id'].tolist()
        val_ids = meta_df[meta_df['split'] == 'val']['image_id'].tolist()
        test_ids = meta_df[meta_df['split'] == 'test']['image_id'].tolist()

        print(f'Training: {len(train_ids)} images, Validation: {len(val_ids)} images, Testing: {len(test_ids)} images.')

        # getting img and lbl paths for each split 
        train_img, train_lbl = get_img_lbl_paths(train_ids)
        val_img, val_lbl = get_img_lbl_paths(val_ids)
        test_img, _ = get_img_lbl_paths(test_ids) # no labels for testing set

        # another check: printing image-label pairs for training and validation splits
        print(f'\n LOADED {len(train_img)} training images.')
        for img, lbl in zip(train_img, train_lbl):
            print(f'Training Image: {img} | Label: {lbl}')
        
        print(f'\n LOADED {len(val_img)} validation images.')
        for img, lbl in zip(val_img, val_lbl):
            print(f'Validation Image: {img} | Label: {lbl}')

        print(f'\nLOADED {len(test_img)} testing images.')
        for img in test_img:
            print(f'Test Image: {img}')



        # defining transforms for training, validation and testing set  --> DATA AUGMENTATION STEP

        # we have to remember that LoadImaged, EnsureChannelFirstd, EnsureTyped, ScaleIntensityd, ResizeWithPadOrCropd, Convert_To_Binary
        # are not the actual augmentation, yet the preprocessing steps. These are as follows:
        # 1. Loading and ensuring the correct data format.
        # 2. Scaling intensities.
        # 3. Resizing the image to the correct spatial dimensions.

        train_transforms = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys=['image', 'label']),
            ScaleIntensityd(keys=['image']),
            RandSpatialCropd(keys =['image', 'label'], roi_size = (128,128,128), random_size = False),
            RandRotate90d(keys=['image', 'label'], prob = 0.5, max_k = 3),
            RandAffined(keys=['image', 'label'], prob = 0.5,  rotate_range=[(-0.1, 0.1)] * 3, scale_range=[(-0.1, 0.1)] * 3, mode=['bilinear', 'nearest']),
            RandGaussianNoised(keys = ['image'], prob = 0.5),
            ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size = (128,128,128)),
        ])

        val_transforms = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys=['image', 'label']),
            ScaleIntensityd(keys=['image']),
            ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size = (128,128,128)),
        ])


        test_transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            ResizeWithPadOrCropd(keys=['image',], spatial_size=(128, 128, 128)),
            EnsureTyped(keys=['image']),
        ])


        # creating datasets for each split 
        train_ds = TotalSeg_Dataset_Tr_Val(
            train_img, 
            train_lbl, 
            cmb_masks = combine_masks, 
            transform= train_transforms
        )

        val_ds = TotalSeg_Dataset_Tr_Val(
            val_img, 
            val_lbl, 
            cmb_masks = combine_masks, 
            transform= val_transforms
        )

        test_ds = TotalSeg_Dataset_Ts(
            test_img,
            transform= test_transforms
        )

        # creating dataloaders for each split 
        train_loader = DataLoader(
            train_ds, 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = num_workers,
            collate_fn = batch_collate_fn_tr_val
        )

        val_loader = DataLoader(
            val_ds, 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = num_workers,
            collate_fn = batch_collate_fn_tr_val
        )

        test_loader = DataLoader(
            test_ds, 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = num_workers,
            collate_fn = batch_collate_fn_test
        )

        return train_loader, val_loader, test_loader

if  __name__ == "__main__":
    base_dir = "Totalsegmentator_dataset_v201" # here, the relative path used 
    meta_csv = "Totalsegmentator_dataset_v201/meta.csv"
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, meta_csv, combine_masks=True)

# WARNING: since the output is big, a good practice is to type in the terminal: 
# python dataloader.py > output_dataloader.txt