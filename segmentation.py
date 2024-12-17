# === segmentation.py ===

import torch
import numpy as np
import nibabel as nib

from monai.transforms import Resize, ScaleIntensity

from model import get_unet_model


# preparing CT scan for segmentation by resizing, normalizing and converting it into proper format 
def preprocess_img(ct_scan, target_size=(128, 128, 128)):
    print(f"Original CT scan array shape: {ct_scan.shape}")  # print not log_message as it is the information for a developer, not a doctor, thus hidden in the code

    # transposing (numpy) array from [Height, Width, Depth] to [Depth, Height, Width] 
    ct_scan = np.transpose(ct_scan, (2, 0, 1))
    print(f"After transpose: {ct_scan.shape}")  # now shape is [Depth, Height, Width]

    # converting (numpy) array to torch tensor
    img_tensor = torch.from_numpy(ct_scan).float()
    print(f"After converting to tensor: {img_tensor.shape}")  # [Depth, Height, Width]

    # adding channel dimension
    img_tensor = img_tensor.unsqueeze(0)  # [Channel, Depth, Height, Width]
    print(f"After unsqueeze(0): {img_tensor.shape}")  

    # scaling intensity to [0,1]
    scaler = ScaleIntensity()
    img_tensor = scaler(img_tensor)
    print(f"After ScaleIntensity: {img_tensor.shape}")  

    # resizing to target size without adding batch dimension
    print(f"Resizing with spatial_size: {target_size}")
    resize = Resize(spatial_size=target_size)
    img_tensor = resize(img_tensor)  # NO unsqueeze as previously 
    print(f"After resize: {img_tensor.shape}")  

    return img_tensor


# takes a preprocessed image tensor and uses a trained model to perform segmentation
def segment_img(model, img_tensor):

    model.eval()
    with torch.no_grad():
        # adding batch size https://www.ultralytics.com/glossary/batch-size
        img_tensor = img_tensor.unsqueeze(0)  # [1, Channel, Depth, Height, Width]
        print(f"Segmenting image with shape: {img_tensor.shape}") 
        
        # moving tensor the the same device as model 
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        print(f"Image tensor moved to the same device: {device} as model.")  
        
        # forward pass of input tensor (img_tensor) through trained model  https://medium.com/@pradeepkr848115/unravelling-the-magic-a-beginners-guide-to-forward-propagation-in-neural-networks-4d247a564528
        seg_out = model(img_tensor)  # [1, num_classes, Depth, Height, Width]
        print(f"Model output shape: {seg_out.shape}")  
        
        # softmax to get probabilities  https://machinelearningmastery.com/softmax-activation-function-with-python/
        seg_probs = torch.softmax(seg_out, dim=1)
        print(f"After softmax: {seg_probs.shape}")  
        
        # class with the highest probability  https://github.com/cjohnson318/til/blob/main/python/argmax-without-numpy.md
        seg_pred = torch.argmax(seg_probs, dim=1)  # [1, Depth, Height, Width]
        print(f"After argmax: {seg_pred.shape}")  
        
    return seg_pred.squeeze(0).cpu().numpy()  # [Depth, Height, Width]


# saving the segmentation
def save_segmentation(seg_out, affine, save_path):

    seg_img = nib.Nifti1Image(seg_out.astype(np.int16), affine)
    nib.save(seg_img, save_path)
    print(f"Segmentation has been saved at: {save_path}")  
