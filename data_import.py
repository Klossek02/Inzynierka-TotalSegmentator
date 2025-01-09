# === data_import.py ===

import nibabel as nib
import numpy as np

# function to validate whether a file is a valid NIfTI file: 
def validate_nifti(file_path):
    try:
        nib.load(file_path)  # source: https://nipy.org/nibabel/gettingstarted.html
        return True
    except Exception:
        return False


# function to load a NIfTI file and return its data and affine transformation matrix:
def load_nifti(file_path):
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        affine = nii.affine
        return data, affine
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {file_path}: {e}")
