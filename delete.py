import os
import glob

base_dir = "Totalsegmentator_dataset_v201"  

for case_dir in sorted(os.listdir(base_dir)):
    if case_dir.startswith('s') and os.path.isdir(os.path.join(base_dir, case_dir)):
        seg_dir = os.path.join(base_dir, case_dir, 'segmentations')
        if os.path.exists(seg_dir):
            combined_path = os.path.join(seg_dir, 'combined_mask.nii.gz')
            if os.path.exists(combined_path):
                print(f"Usuwam: {combined_path}")
                os.remove(combined_path)
