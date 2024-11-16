# === data_import.py ===

import nibabel as nib

def validate_nifti(file_path):
    try:
        nib.load(file_path)
        return True
    except Exception:
        return False

def load_nifti(file_path):
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        affine = nii.affine
        return data, affine
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {file_path}: {e}")
