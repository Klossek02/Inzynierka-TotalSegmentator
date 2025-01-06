# === data_import.py ===

import nibabel as nib
import numpy as np

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


def get_scan_orientation(affine):
    return nib.orientations.io_orientation(affine)


def reorient_scan(data, affine, desired_ornt =('R', 'A', 'S'), logger=None):
    try:
        current_ornt = get_scan_orientation(affine)
        desired_ornt_matrix = nib.orientations.axcodes2ornt(desired_ornt)
        transform = nib.orientations.ornt_transform(current_ornt, desired_ornt_matrix)
        reoriented_data = nib.orientations.apply_orientation(data, transform)
        new_affine = affine.copy()
        new_affine = nib.orientations.inv_ornt_aff(transform, data.shape)
        new_affine = np.dot(new_affine, affine)
        
        if logger:
            logger("Scan data has been reoriented to the desired orientation.")
        
        return reoriented_data, new_affine
    except Exception as e:
        if logger:
            logger(f"Error in reorient_scan: {e}")
        raise
