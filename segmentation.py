# === segmentation.py ===

import nibabel as nib
import numpy as np
import torch

from monai.transforms import Resize, ScaleIntensity
from totalsegmentator.python_api import totalsegmentator


__name__ = '__main__'  

def segment_img_with_TS(seg_in):
    """  
    Method for segmenting CT scan with TotalSegmentator model.
    Source: https://github.com/wasserth/TotalSegmentator/blob/master/README.md?fbclid=IwZXh0bgNhZW0CMTAAAR10eO8vUynsUgYRJ2BqMUkMXDqHeUTa1r5RjxZrnJrmfHXOuecCBASiL1I_aem_qLQKfOT9tQP_LVGT_npG5A
    """

    if __name__ == '__main__':
        seg_out = totalsegmentator(seg_in, fast=True)
        return seg_out


def save_segmentation(seg_out, affine, save_path):
    """  
    Method for saving the segmentation.
    """ 

    seg_img = nib.Nifti1Image(seg_out.astype(np.int16), affine)  # source: https://nipy.org/nibabel/reference/nibabel.nifti1.html
    nib.save(seg_img, save_path)  # source: https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
    print(f"Segmentation has been saved at: {save_path}")