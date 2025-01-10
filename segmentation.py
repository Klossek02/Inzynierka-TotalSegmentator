# === segmentation.py ===

import nibabel as nib
import numpy as np
import torch

from monai.transforms import Resize, ScaleIntensity
from totalsegmentator.python_api import totalsegmentator


__name__ = '__main__'

def preprocess_img(ct_scan, target_size=(128, 128, 128)):
    """ 
    Method for preparing CT scan for segmentation by resizing, normalizing and converting it into proper format.
    """

    print(f"Original CT scan array shape: {ct_scan.shape}")  # print not log_message as it is the information for a developer, not a doctor, thus hidden in the code

    # transposing (numpy) array from [Height, Width, Depth] to [Depth, Height, Width]: 
    ct_scan = np.transpose(ct_scan, (2, 0, 1)) # source: https://numpy.org/doc/2.1/reference/generated/numpy.transpose.html
    print(f"After transpose: {ct_scan.shape}")  # now shape is [Depth, Height, Width]

    # converting (numpy) array to torch tensor: 
    img_tensor = torch.from_numpy(ct_scan).float() # source: https://pytorch.org/docs/stable/generated/torch.from_numpy.html
    print(f"After converting to tensor: {img_tensor.shape}")  # [Depth, Height, Width]

    # adding channel dimension:
    img_tensor = img_tensor.unsqueeze(0)  # [Channel, Depth, Height, Width] ; source: https://pytorch.org/docs/main/generated/torch.unsqueeze.html
    print(f"After unsqueeze(0): {img_tensor.shape}")  

    # scaling intensity to [0,1]: 
    scaler = ScaleIntensity() # source: https://github.com/Project-MONAI/MONAI/blob/main/monai/transforms/intensity/array.py
    img_tensor = scaler(img_tensor) 
    print(f"After ScaleIntensity: {img_tensor.shape}")  

    # resizing to target size without adding batch dimension: 
    print(f"Resizing with spatial_size: {target_size}")
    resize = Resize(spatial_size=target_size)
    img_tensor = resize(img_tensor)  # NO unsqueeze as previously 
    print(f"After resize: {img_tensor.shape}")  

    return img_tensor


def segment_img(model, img_tensor):
    """   
    Method that takes a preprocessed image tensor and uses a trained model to perform segmentation.
    """

    model.eval()
    with torch.no_grad():
        # adding batch size; source: https://www.ultralytics.com/glossary/batch-size
        img_tensor = img_tensor.unsqueeze(0)  # [1, Channel, Depth, Height, Width]
        print(f"Segmenting image with shape: {img_tensor.shape}") 
        
        # moving tensor the the same device as the model: 
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        print(f"Image tensor moved to the same device: {device} as model.")  
        
        # forward pass of input tensor (img_tensor) through trained model; source:  https://medium.com/@pradeepkr848115/unravelling-the-magic-a-beginners-guide-to-forward-propagation-in-neural-networks-4d247a564528
        seg_out = model(img_tensor)  # [1, num_classes, Depth, Height, Width]
        print(f"Model output shape: {seg_out.shape}")  
        
        # softmax to get probabilities; source:  https://machinelearningmastery.com/softmax-activation-function-with-python/
        seg_probs = torch.softmax(seg_out, dim=1)
        print(f"After softmax: {seg_probs.shape}")  
        
        # class with the highest probability; source:  https://github.com/cjohnson318/til/blob/main/python/argmax-without-numpy.md
        seg_pred = torch.argmax(seg_probs, dim=1)  # [1, Depth, Height, Width]
        print(f"After argmax: {seg_pred.shape}")  
        
    return seg_pred.squeeze(0).cpu().numpy()  # [Depth, Height, Width]


def save_segmentation(seg_out, affine, save_path):
    """  
    Method for saving the segmentation.
    """ 

    seg_img = nib.Nifti1Image(seg_out.astype(np.int16), affine)  # source: https://nipy.org/nibabel/reference/nibabel.nifti1.html
    nib.save(seg_img, save_path)  # source: https://bic-berkeley.github.io/psych-214-fall-2016/saving_images.html
    print(f"Segmentation has been saved at: {save_path}")  


def segment_img_with_TS(seg_in):
    """  
    Method for segmenting CT scan with TotalSegmentator model.
    Source: https://github.com/wasserth/TotalSegmentator/blob/master/README.md?fbclid=IwZXh0bgNhZW0CMTAAAR10eO8vUynsUgYRJ2BqMUkMXDqHeUTa1r5RjxZrnJrmfHXOuecCBASiL1I_aem_qLQKfOT9tQP_LVGT_npG5A
    """

    if __name__ == '__main__':
        seg_out = totalsegmentator(seg_in, fast=True)
        return seg_out