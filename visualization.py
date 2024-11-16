# === visualization.py ===

import os
import numpy as np
import cv2
import nibabel as nib

from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# https://radiopaedia.org/articles/labelled-imaging-anatomy-cases
# https://pro.boehringer-ingelheim.com/us/ipfradiologyrounds/hrct-primer/image-reconstruction


# converts NifTi file into saggital, coronal and axial 2D slices, resizing them to a target size (the one fitting to the window)
def convert_img_slices(file_path, target_size=(400, 300, 128), logger=None):
    try:
        if logger:
            logger(f"Loading NIfTI file: {file_path}.")
            
        nii = nib.load(file_path) # https://nipy.org/nibabel/gettingstarted.html
        ct_scan = nii.get_fdata()
        affine = nii.affine
        ct_scan = ct_scan.astype(np.float32)
        
        D, H, W = ct_scan.shape # D - Depth, H - Height, W - Width
        new_W, new_H, new_D = target_size  
        
        out_array_sagittal, out_array_coronal, out_array_axial = [], [], []
    
        if logger:
            logger("Converting slices...")

        # sagittal slices
        for i in range(new_D):
            orig_i = int(i * D / new_D)
            slice_2D = ct_scan[orig_i, :, :]
            slice_2D_resized = cv2.resize(slice_2D, (new_W, new_H)) # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
            slice_2D_normalized = cv2.normalize(slice_2D_resized, None, 0, 255, cv2.NORM_MINMAX) # https://www.geeksforgeeks.org/normalize-an-image-in-opencv-python/
            slice_2D_uint8 = slice_2D_normalized.astype(np.uint8)
            qimage = QImage(slice_2D_uint8.data.tobytes(), slice_2D_uint8.shape[1], slice_2D_uint8.shape[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage) # https://doc.qt.io/qt-6/qpixmap.html
            out_array_sagittal.append(pixmap)
            if logger and i % 10 == 0:
                logger(f"Sagittal slice {i+1}/{new_D} has been converted.")


        # coronal slices
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

        # axial slices
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
            logger("Finished converting all slices.")

        return out_array_sagittal, out_array_coronal, out_array_axial, affine

    except Exception as e:
        raise ValueError(f"Failed to convert NIfTI file {file_path}: {e}")

# displays single slice within provided QLabel and QPixmap (contianing this slice)
def display_single_slice(label: QLabel, pixmap: QPixmap):
    scaled_ = pixmap.scaled( # https://doc.qt.io/qt-6/qt.html
        label.width(),
        label.height(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    label.setPixmap(scaled_)
