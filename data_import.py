# === data_import.py ===

import nibabel as nib
import numpy as np
 
def validate_nifti(file_path):
    """ 
    Method for validating whether a file is a valid NIfTI file.
    """

    try:
        nib.load(file_path)  # source: https://nipy.org/nibabel/gettingstarted.html
        return True
    except Exception:
        return False


def load_nifti(file_path):
    """  
    Method for loading a NIfTI file and return its data and affine transformation matrix.
    """

    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        affine = nii.affine
        return data, affine
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {file_path}: {e}")
