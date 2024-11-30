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
    RandSpatialCropd, RandRotate90d, RandAffined, RandZoomd, RandAxisFlipd,
    RandGaussianNoised, Rand3DElasticd, ResizeWithPadOrCropd, ConcatItemsd
)
from monai.data import Dataset, CacheDataset

# We are working on the classification of various body parts, grouped into 117 anatomical structures. 
# Our aim is to simplify things by combining masks for these structures into single binary mask (eg. lung lobes into whole lung)
# the exact grouping is based on the README here: https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TotalSegmentator_v2.md

# Note that for some classes the names differ from the standarized anatomical names. The mapping can be found here: https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file

main_classes_CT = {
"skeleton": [
'skull', 'clavicula_left', 'clavicula_right', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'sternum',
'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8',
'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4',
'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12',
'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1',
'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3',
'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9', 'vertebrae_T10',
'vertebrae_T11', 'costal_cartilages', 'vertebrae_T12', 'hip_left', 'hip_right', 'sacrum', 'femur_left', 'femur_right'],

"cardiovascular": [
'common_carotid_artery_left', 'common_carotid_artery_right', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right',
'subclavian_artery_left', 'subclavian_artery_right', 'brachiocephalic_trunk', 'superior_vena_cava', 'pulmonary_vein',
'atrial_appendage_left', 'aorta', 'heart', 'portal_vein_and_splenic_vein', 'inferior_vena_cava', 'iliac_artery_left',
'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right'],

"gastrointestinal":[
'esophagus', 'stomach', 'duodenum', 'small_bowel', 'colon', 'urinary_bladder'],


"muscles":[
'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right', 'gluteus_minimus_left', 
'gluteus_minimus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_maximus_left', 
'gluteus_maximus_right'],

"others":[
'brain', 'spinal_cord', 'thyroid_gland', 'trachea', 'lung_upper_lobe_left', 'lung_upper_lobe_right', 
'lung_middle_lobe_right', 'lung_lower_lobe_left', 'lung_lower_lobe_right','adrenal_gland_left', 
'adrenal_gland_right', 'spleen', 'liver', 'gallbladder', 'kidney_left', 'kidney_right', 'kidney_cyst_left',
'kidney_cyst_right', 'pancreas', 'prostate']

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
        

    def __len__(self): # returns tot. number of images 
        return len(self.img_paths)

    def __getitem__(self, idx): # loading image and label paths for given index 
        img_path = self.img_paths[idx]  # get img file path for current index 
        lbl_path = self.lbl_paths[idx]  # -=- for lbl 

        try:
            # since labels have a default path defined as: "Totalsegmentator_dataset_v201\s****\segmentations",
            # we need to ensure that we can access the *.nii.gz files from that directory as these will be necessary for model training, 
            # not the folder 'segmentations' itself. 
        
            lbl_files = glob.glob(os.path.join(lbl_path, '*.nii.gz')) 
            print('lbl_files:', lbl_files)

            # checking if the label file exists 
            if len(lbl_files) == 0:
                raise FileNotFoundError(f'ERROR: missing label for {img_path}. Cannot work out file type.')

            if self.cmb_masks: # case when need to combine mask into single label 
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


    # this function combines individual masks into a single binary mask. 
    def combine_masks(self, lbl_dir): 
        out_path = os.path.join(lbl_dir, 'combined_mask.nii.gz')  # https://docs.python.org/3/library/os.path.html

        if not os.path.exists(out_path):  # check if the combined mask already exists to avoid recalculation
            mask_files = glob.glob(os.path.join(lbl_dir, '*.nii.gz')) # get all nii.gz files (mask files)
            combined_mask = None

            # assiging uniqe label, numeric one, to each class in order to distinguish it 
            class_labels = {class_name: idx + 1 for idx, class_name in enumerate(main_classes_CT)}
            
            # now, iterate through all anatomical groups and combine their structure masks 
            for class_group, structures in main_classes_CT.items():
                for struct in structures:
                    structure_mask_file = os.path.join(lbl_dir, f'{struct}.nii.gz') 

                    # checking if this specific structure's mask exists 
                    if os.path.exists(structure_mask_file):
                        mask = nib.load(structure_mask_file).get_fdata() # https://nipy.org/nibabel/images_and_memory.html - load mask data

                        if combined_mask is None: 
                            combined_mask = np.zeros_like(mask) # https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html - generating empty mask

                        combined_mask[mask > 0] = class_labels[class_group] # any non-zero value in the current mask will be assigned to the class label

            # covert the combined mask into NIfTI format and save it
            affine = nib.load(mask_files[0]).affine
            combined_mask_img = nib.Nifti1Image(combined_mask.astype(np.uint8), affine=affine)
            nib.save(combined_mask_img, out_path)


        return out_path # return the path to the combined mask file


# class for loading test images 
class TotalSeg_Dataset_Ts(Dataset):

    def __init__(self,img_paths, transform = None): # initialization the dataset with image paths and any transforms 
        self.img_paths = img_paths 
        self.transform = transform 

    def __len__(self): # returns tot. number of images 
        return len(self.img_paths)

    def __getitem__(self, idx): # loading image paths for given index 
        img_path = self.img_paths[idx]   # get img file path for current index 
        img = nib.load(img_path).get_fdata()  # load img data with NIfTI format

    
        data = {'image': img } # organizing data into a dictionary (in case we want to add more data later)

        if self.transform is not None: # if transform is provided 
            data = self.transform(data) # applying any transform 

        return data


# class for converting masks to binary - 0 or 1
# it converts a non-zero label values to 1, turning multi-class segmentation into binary segmentation (presence vs. absence)
# https://medium.com/@mhamdaan/multi-class-semantic-segmentation-with-u-net-pytorch-ee81a66bba89
# https://www.sciencedirect.com/science/article/pii/S0167839621000182

class Convert_To_Binary: 
    def __call__(self, key): # key - dictionary with keys 'img' and 'label'
        key['label'] = torch.where(key['label']> 0, 1, 0)
        return key 


# in this step we create a collate function which handles "None" batches. 
    # this custom function returns either dict ('image': image_tensor of size shape [1, 277, 277, 95], 'label': label_tensor of size shape [117, 277, 277, 95]) or None in case 
    # all samples are invalid 
    # https://lukesalamone.github.io/posts/custom-pytorch-collate/
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/_utils/collate.html#default_collate
def batch_collate_fn(batch):  # here we called arg 'batch' instead of data

    # first, let us filter out invalid samples (samples that are None or have 'image' or 'label' as None)
    filtered_batch = [
        sample for sample in batch
        if sample is not None 
        ]

    if len(filtered_batch) == 0: # if not valid samples present 
        return None 

    # then, let us group the samples by 'image' and 'label'
    images = [sample['image'] for sample in filtered_batch]
    labels =  [sample['label'] for sample in filtered_batch]

    # finally, let us stack the 'image' and 'label' tensors
    batched_images = torch.stack(images, dim=0)  # shape: [batch_size, 1, 277, 277, 95]
    batched_labels = torch.stack(labels, dim=0)  # shape: [batch_size, 117, 277, 277, 95]

    return {'image': batched_images, 'label': batched_labels}



# function that loads and preapres data for training, validation and testing 
"""
    base_dir: base directory where img data is stored 
    meta_csv: path to csv file containing meta information for train/test/val splits
    cmb_masks: if True, combines individual masks into a single binary mask. If False, otherwise
    batch_size: no. of samples per batch 
    num_workers: no. of workers for loading data 
 """

def get_dataloaders(base_dir, meta_csv, combine_masks = True, batch_size = 1, num_workers = 1):
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

        # Another check: printing image-label pairs for training and validation splits
        print(f'\n LOADED {len(train_img)} training images.')
        for img, lbl in zip(train_img, train_lbl):
            print(f'Training Image: {img} | Label: {lbl}')
        
        print(f'\n LOADED {len(val_img)} validation images.')
        for img, lbl in zip(val_img, val_lbl):
            print(f'Validation Image: {img} | Label: {lbl}')

        #  printing image pairs for testing split
        print(f"\n Number of test images: {len(test_img)}")

        for img in test_img:
            print(f"Image: {img}")


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
            #RandZoomd(keys=['image', 'label'], prob = 0.7, min_zoom=1.1 , max_zoom = 1.2, mode=['trilinear', 'nearest'], align_corners=True),
            #RandAxisFlipd(keys = ['image', 'label'], prob = 0.5),
            RandGaussianNoised(keys = ['image'], prob = 0.5),
            #Rand3DElasticd(keys=['image, label'], prob = 0.2, sigma_range = (5, 5, 5), magnitude_range= (1.1, 2), mode = ['bilinear, nearest']),
            ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size = (128,128,128)),
            Convert_To_Binary(),
        ])

        val_transforms = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys=['image', 'label']),
            ScaleIntensityd(keys=['image']),
            ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size = (128,128,128)),
            Convert_To_Binary(),
        ])


        test_transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            ResizeWithPadOrCropd(keys=['image'], spatial_size=(128, 128, 128)),  
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
            collate_fn = batch_collate_fn
        )

        val_loader = DataLoader(
            val_ds, 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = num_workers,
            collate_fn = batch_collate_fn
        )

        test_loader = DataLoader(
            test_ds, 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = num_workers,
            collate_fn = batch_collate_fn
        )

        return train_loader, val_loader, test_loader

if  __name__ == "__main__":
    base_dir = "Totalsegmentator_dataset_v201" # here, the relative path used 
    meta_csv = "Totalsegmentator_dataset_v201/meta.csv"
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, meta_csv, combine_masks=True)

# WARNING: since the output is big, a good practice is to type in the terminal: 
# python dataloader.py > output.txt 