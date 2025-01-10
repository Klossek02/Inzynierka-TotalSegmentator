# === visualization.py ===

import cv2
import nibabel as nib
import numpy as np
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel

# https://radiopaedia.org/articles/labelled-imaging-anatomy-cases
# https://pro.boehringer-ingelheim.com/us/ipfradiologyrounds/hrct-primer/image-reconstruction


def convert_img_slices(file_path, target_size=(400, 300, 128), logger=None):
    """ 
    Method for converting NifTi file into saggital, coronal and axial 2D slices, resizing them to a target size (the one fitting to the window).
    """
    
    try:
        if logger:
            logger(f"Loading NIfTI file: {file_path}.")
            
        nii = nib.load(file_path) # source: https://nipy.org/nibabel/gettingstarted.html
        ct_scan = nii.get_fdata() # getting 3D array
        affine = nii.affine # affine transformation matrix 
        ct_scan = ct_scan.astype(np.float32) # converting data to float32 
        
        D, H, W = ct_scan.shape # dimensions of the CT scan: D - depth, H - height, W - width
        new_W, new_H, new_D = target_size  # target dimensions for resizing slices
        
        out_array_sagittal, out_array_coronal, out_array_axial = [], [], []
    
        if logger:
            logger("Converting slices...")

        # processing sagittal slices:
        for i in range(new_D):
            orig_i = int(i * D / new_D) # mapping the slice index to the original depth
            slice_2D = ct_scan[orig_i, :, :] # extracting 2D slice
            slice_2D_resized = cv2.resize(slice_2D, (new_W, new_H)) # resizing the slicel source: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
            slice_2D_normalized = cv2.normalize(slice_2D_resized, None, 0, 255, cv2.NORM_MINMAX) # normalize values to 0-255; source: https://www.geeksforgeeks.org/normalize-an-image-in-opencv-python/
            slice_2D_uint8 = slice_2D_normalized.astype(np.uint8)
            qimage = QImage(slice_2D_uint8.data.tobytes(), slice_2D_uint8.shape[1], slice_2D_uint8.shape[0], QImage.Format_Grayscale8) # converting to uint8; source: https://doc.qt.io/qt-6/qpixmap.html
            pixmap = QPixmap.fromImage(qimage) # converting QImage to QPixmap; source: https://doc.qt.io/qt-6/qimage.html https://doc.qt.io/qt-6/qpixmap.html
            out_array_sagittal.append(pixmap) # adding pixmap to sagittal array
            if logger and i % 10 == 0:
                logger(f"Sagittal slice {i+1}/{new_D} has been converted.")


        # processing coronal slices:
        for i in range(new_H):
            orig_i = int(i * H / new_H)
            slice_2D = ct_scan[:, orig_i, :]
            slice_2D_resized = cv2.resize(slice_2D, (new_W, new_D))
            slice_2D_normalized = cv2.normalize(slice_2D_resized, None, 0, 255, cv2.NORM_MINMAX)
            slice_2D_uint8 = slice_2D_normalized.astype(np.uint8)
            qimage = QImage(slice_2D_uint8.data.tobytes(), slice_2D_uint8.shape[1], slice_2D_uint8.shape[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            out_array_coronal.append(pixmap)
            if logger and i % 10 == 0:
                logger(f"Coronal slice {i+1}/{new_H} has been converted.")

        # processing axial slices:
        for i in range(new_W):
            orig_i = int(i * W / new_W)
            slice_2D = ct_scan[:, :, orig_i]
            slice_2D_resized = cv2.resize(slice_2D, (new_H, new_D))
            slice_2D_normalized = cv2.normalize(slice_2D_resized, None, 0, 255, cv2.NORM_MINMAX)
            slice_2D_uint8 = slice_2D_normalized.astype(np.uint8)
            qimage = QImage(slice_2D_uint8.data.tobytes(), slice_2D_uint8.shape[1], slice_2D_uint8.shape[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            out_array_axial.append(pixmap)
            if logger and i % 10 == 0:
                logger(f"Axial slice {i+1}/{new_W} has been converted.")

        if logger:
            logger("Converting all slices finished.")

        return out_array_sagittal, out_array_coronal, out_array_axial, affine

    except Exception as e:
        raise ValueError(f"Failed to convert NIfTI file {file_path}: {e}")


def display_single_slice(label: QLabel, pixmap: QPixmap):
    """   
    Method for displaying a single slice within provided QLabel and QPixmap (contianing this slice).
    """

    scaled_ = pixmap.scaled( # scaling the pixmap to fit within QLabel, simultaneusly preserving aspect ratio; source: https://doc.qt.io/qt-6/qt.html
        label.width(),
        label.height(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    label.setPixmap(scaled_)


def convert_seg_slices(seg_data, target_size=(400, 300, 128), logger=None):
    """ 
    Method for converting segmentation slices into sagittal, coronal and axial ones.
    """ 

    D, H, W = seg_data.shape
    new_W, new_H, new_D = target_size

    out_sagittal = []
    out_coronal = []
    out_axial = []

    if logger:
        logger("Converting segmentation slices...")

    # sagittal
    for i in range(new_D):
        orig_i = int(i * D / new_D)
        slice_2D = seg_data[orig_i, :, :]  
        slice_resized = cv2.resize(slice_2D, (new_W, new_H), interpolation=cv2.INTER_NEAREST) # source for entire cv2 library: https://opencv.org/, https://pypi.org/project/opencv-python/
        pixmap_mask = mask_to_qpixmap(slice_resized)
        out_sagittal.append(pixmap_mask)

    # coronal
    for i in range(new_H):
        orig_i = int(i * H / new_H)
        slice_2D = seg_data[:, orig_i, :]  
        slice_resized = cv2.resize(slice_2D, (new_W, new_D), interpolation=cv2.INTER_NEAREST)
        pixmap_mask = mask_to_qpixmap(slice_resized)
        out_coronal.append(pixmap_mask)

    # axial
    for i in range(new_W):
        orig_i = int(i * W / new_W)
        slice_2D = seg_data[:, :, orig_i]  
        slice_resized = cv2.resize(slice_2D, (new_H, new_D), interpolation=cv2.INTER_NEAREST)
        pixmap_mask = mask_to_qpixmap(slice_resized)
        out_axial.append(pixmap_mask)

    if logger:
        logger("Converting segmentation slices finished.")
    return out_sagittal, out_coronal, out_axial



def mask_to_qpixmap(mask_2d: np.ndarray) -> QPixmap: 
    """ 
    # Method for converting segmentation mask (2D) to QPixmap.
    """

    mask_2d_uint8 = mask_2d.astype(np.uint8)  # source: https://numpy.org/devdocs/reference/generated/numpy.astype.html
    h, w = mask_2d_uint8.shape
    qimg = QImage(
        mask_2d_uint8.data.tobytes(),  # source: https://numpy.org/devdocs/reference/generated/numpy.ndarray.tobytes.html
        w, h,
        w,      # bytes per line = width * no. of channels (here 1 in grayscale)
        QImage.Format_Grayscale8
    )
    return QPixmap.fromImage(qimg)



def overlay_slices(base_slice: QPixmap, mask_slice: QPixmap,
                   alpha=0.4, class_to_color=None) -> QPixmap:

    """ 
    # Method for overlaying segmentation slices on provided (base) images with specified transparency.
    
    """

    # converting QPixmap to QImage and then to numpy array (grayscale):
    base_img = base_slice.toImage().convertToFormat(QImage.Format_Grayscale8)
    w, h = base_img.width(), base_img.height()
    ptr_base = base_img.bits()
    ptr_base.setsize(base_img.byteCount())
    arr_base = np.frombuffer(ptr_base, np.uint8).reshape((h, w))  # 2D grayscale

    mask_img = mask_slice.toImage().convertToFormat(QImage.Format_Grayscale8)
    w_m, h_m = mask_img.width(), mask_img.height()
    ptr_mask = mask_img.bits()
    ptr_mask.setsize(mask_img.byteCount())
    arr_mask = np.frombuffer(ptr_mask, np.uint8).reshape((h_m, w_m))  # 2D with labels

    if arr_base.shape != arr_mask.shape:
        # resizing mask to match base image size:
        arr_mask = cv2.resize(arr_mask, (w, h), interpolation=cv2.INTER_NEAREST) # source: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

    # converting base to BGR:
    base_bgr = cv2.cvtColor(arr_base, cv2.COLOR_GRAY2BGR)  # source: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

    # creating overlay BGR by filling pixels (based on the label)
    overlay_bgr = np.zeros_like(base_bgr, dtype=np.uint8)  # source: https://numpy.org/doc/2.1/reference/generated/numpy.zeros_like.html
    unique_lbls = np.unique(arr_mask)  # source: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    for lbl in unique_lbls:
        if lbl == 0:
            continue
        color_bgr = (255, 0, 0)  # blue by default (otherwise, some color to choose that the user may like)
        if class_to_color and lbl in class_to_color:
            color_hex = class_to_color[lbl]
            color_bgr = _hex_to_bgr(color_hex)
        overlay_bgr[arr_mask == lbl] = color_bgr

    # blending overlay and base (similar to blending images logic):
    blended_bgr = cv2.addWeighted( # https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
        src1=base_bgr, alpha=1.0,
        src2=overlay_bgr, beta=alpha,
        gamma=0.0
    )
    # converting to RGB: 
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)  # source: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    qimg_final = QImage( # source: https://doc.qt.io/qt-6/qimage.html
        blended_rgb.data.tobytes(),
        w, h,
        3 * w,
        QImage.Format_RGB888
    )
    return QPixmap.fromImage(qimg_final)


def _hex_to_bgr(hex_color: str):
    """ 
    # Method for converting hex color to BGR (Blue, Green, Red) format.
    """

    hex_color = hex_color.lstrip('#') # remove leading # from hex color 
    if len(hex_color) != 6:
        return (255, 0, 0)  # default blue color if the hex color is not valid.'

    # now, we extract red, green, blue values from hex color (hex == base 16): 
    r = int(hex_color[0:2], 16) # RED: converting first two hex characters to int 
    g = int(hex_color[2:4], 16) # GREEN: converting next two hex characters to int
    b = int(hex_color[4:6], 16) # BLUE: converting last two hex characters to int
    return (b, g, r)


def get_scan_orientation(affine):

    """ 
    Method for obtaining scan orientation using its affine matrix.
    """
    return nib.orientations.io_orientation(affine)


def reorient_scan(data, affine, desired_ornt =('R', 'A', 'S'), logger=None): # using nibabel for reorientation

    """   
    Method for reorienting scan to match the desired orientation.
    """

    try:
        current_ornt = get_scan_orientation(affine) # getting current scan orientation
        desired_ornt_matrix = nib.orientations.axcodes2ornt(desired_ornt) # converting desired orientation to matrix; source: https://nipy.org/nibabel/reference/nibabel.orientations.html
        transform = nib.orientations.ornt_transform(current_ornt, desired_ornt_matrix)  # computing transformation to align current orientation to the desired one
        reoriented_data = nib.orientations.apply_orientation(data, transform) # applying orientation tranformation 
        new_affine = affine.copy() # adjusting affine matrix to match desired orientation (or in other words, to reflect a new one)
        new_affine = nib.orientations.inv_ornt_aff(transform, data.shape)
        new_affine = np.dot(new_affine, affine)
        
        if logger:
            logger("Scan data has been reoriented to the desired orientation.")
        
        return reoriented_data, new_affine
    except Exception as e:
        if logger:
            logger(f"Error in reorient_scan: {e}")
        raise
