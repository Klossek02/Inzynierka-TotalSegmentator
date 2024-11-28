# === test_application.py ===

import unittest
import os
import numpy as np
import nibabel as nib
import torch
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage
import sys

import data_import
import visualization
import segmentation
import model
import demo

class TestDataImport(unittest.TestCase):
    def test_validate_nifti_valid(self):
        # Create a temporary NIfTI file
        data = np.zeros((10, 10, 10))
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_valid_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # Test validate_nifti
        self.assertTrue(data_import.validate_nifti(temp_file))

        # Clean up
        os.remove(temp_file)

    def test_validate_nifti_invalid(self):
        # Create a non-NIfTI file
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # Test validate_nifti
        self.assertFalse(data_import.validate_nifti(temp_file))

        # Clean up
        os.remove(temp_file)

    def test_load_nifti_valid(self):
        # Create a temporary NIfTI file
        data = np.random.rand(10, 10, 10)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_valid_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # Test load_nifti
        loaded_data, loaded_affine = data_import.load_nifti(temp_file)
        np.testing.assert_array_equal(data, loaded_data)
        np.testing.assert_array_equal(affine, loaded_affine)

        # Clean up
        os.remove(temp_file)

    def test_load_nifti_invalid(self):
        # Create a non-NIfTI file
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # Test load_nifti, should raise ValueError
        with self.assertRaises(ValueError):
            data_import.load_nifti(temp_file)

        # Clean up
        os.remove(temp_file)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Initialize QApplication if not already running
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

    def test_convert_img_slices(self):
        # Create a temporary NIfTI file
        data = np.random.rand(50, 50, 50)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # Test convert_img_slices
        scan_list_sagittal, scan_list_coronal, scan_list_axial, loaded_affine = visualization.convert_img_slices(temp_file)

        # Check that the lists have the expected number of slices
        self.assertEqual(len(scan_list_sagittal), 128)
        self.assertEqual(len(scan_list_coronal), 300)
        self.assertEqual(len(scan_list_axial), 400)

        # Clean up
        os.remove(temp_file)

    def test_convert_img_slices_invalid_file(self):
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # Should raise ValueError
        with self.assertRaises(ValueError):
            visualization.convert_img_slices(temp_file)

        # Clean up
        os.remove(temp_file)

    def test_display_single_slice(self):
        label = QLabel()
        label.setFixedSize(400, 300)

        # Create a dummy pixmap
        data = np.random.rand(300, 400)
        data_uint8 = (data * 255).astype(np.uint8)
        qimage = QImage(data_uint8.data, data_uint8.shape[1], data_uint8.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # Test display_single_slice
        visualization.display_single_slice(label, pixmap)

        # Check that the label's pixmap is set
        self.assertIsNotNone(label.pixmap())

class TestSegmentation(unittest.TestCase):
    def test_preprocess_img(self):
        # Create dummy CT scan data
        data = np.random.rand(128, 128, 128)

        # Call preprocess_img
        img_tensor = segmentation.preprocess_img(data)

        # Check the shape of the output tensor
        self.assertEqual(img_tensor.shape, (1, 128, 128, 128))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))

    def test_segment_img(self):
        # Create dummy model
        model_instance = model.get_unet_model(num_classes=2, in_channels=1)

        # Create dummy img_tensor
        img_tensor = torch.rand((1, 128, 128, 128))

        # Call segment_img
        seg_pred = segmentation.segment_img(model_instance, img_tensor)

        # Check the shape of the output
        self.assertEqual(seg_pred.shape, (128, 128, 128))
        self.assertTrue(isinstance(seg_pred, np.ndarray))

    def test_save_segmentation(self):
        # Create dummy segmentation output
        seg_out = np.random.randint(0, 2, (128, 128, 128))
        affine = np.eye(4)
        save_path = 'temp_segmentation.nii.gz'

        # Call save_segmentation
        segmentation.save_segmentation(seg_out, affine, save_path)

        # Check that file exists
        self.assertTrue(os.path.exists(save_path))

        # Optionally, load and check the content
        loaded_seg = nib.load(save_path).get_fdata()
        np.testing.assert_array_equal(seg_out, loaded_seg)

        # Clean up
        os.remove(save_path)

class TestModel(unittest.TestCase):
    def test_get_unet_model(self):
        num_classes = 2
        in_channels = 1
        net = model.get_unet_model(num_classes=num_classes, in_channels=in_channels)
        self.assertIsNotNone(net)
        self.assertEqual(net.in_channels, in_channels)
        self.assertEqual(net.out_channels, num_classes)
        # Check that net is an instance of UNet
        from monai.networks.nets import UNet
        self.assertIsInstance(net, UNet)

 

class TestDemo(unittest.TestCase):
    def test_convert_to_stl(self):
        # Create a dummy numpy array
        data = np.zeros((10, 10, 10))
        data[3:7, 3:7, 3:7] = 1  # Create a cube inside
        out_path = 'temp_mesh.stl'

        # Call convert_to_stl
        demo.convert_to_stl(data, out_path)

        # Check that file exists
        self.assertTrue(os.path.exists(out_path))

        # Optionally, load the STL and check contents
        from stl import mesh as stl_mesh
        obj_3d = stl_mesh.Mesh.from_file(out_path)
        self.assertIsNotNone(obj_3d)

        # Clean up
        os.remove(out_path)

    def test_convert_to_stl_empty_data(self):
        # Create an empty numpy array
        data = np.zeros((10, 10, 10))
        out_path = 'temp_mesh_empty.stl'

        # Call convert_to_stl
        demo.convert_to_stl(data, out_path)

        # Check that file does not exist (since the data is empty)
        self.assertFalse(os.path.exists(out_path))

if __name__ == '__main__':
    unittest.main()
