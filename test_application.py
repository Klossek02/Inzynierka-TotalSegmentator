# === test_application.py ===

import unittest
import os
import sys
import torch
import time
import numpy as np
import nibabel as nib

from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from unittest.mock import patch, MagicMock

import data_import
import visualization
import segmentation
import model
import demo
from GUI import MedicalImageViewer
from segmentation import preprocess_img, segment_img
from model import get_unet_model
from visualization import display_single_slice

# All tests have been written preserving TDD principles

class TestDataImport(unittest.TestCase):
    def test_validate_nifti_valid(self):

        # Given 
        data = np.zeros((10, 10, 10))
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_valid_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # When 
        result = data_import.validate_nifti(temp_file)

        # Then
        self.assertTrue(result)

        os.remove(temp_file)

    def test_validate_nifti_invalid(self):

        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('ERROR: not a NIfTI file.')

        # When 
        result = data_import.validate_nifti(temp_file)

        # Then
        self.assertFalse(result)

        os.remove(temp_file)

    def test_load_nifti_valid(self):

        # Given
        data = np.random.rand(10, 10, 10)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_valid_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # When
        loaded_data, loaded_affine = data_import.load_nifti(temp_file)

        # Then
        np.testing.assert_array_equal(data, loaded_data)
        np.testing.assert_array_equal(affine, loaded_affine)

        os.remove(temp_file)

    def test_load_nifti_invalid(self):

        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # When, Then
        with self.assertRaises(ValueError): 
            data_import.load_nifti(temp_file)

        os.remove(temp_file)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

    def test_convert_img_slices(self):

        # Given
        data = np.random.rand(50, 50, 50)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # When
        scan_list_sagittal, scan_list_coronal, scan_list_axial, loaded_affine = visualization.convert_img_slices(temp_file)

        # Then
        self.assertEqual(len(scan_list_sagittal), 128)
        self.assertEqual(len(scan_list_coronal), 300)
        self.assertEqual(len(scan_list_axial), 400)

        os.remove(temp_file)

    def test_convert_img_slices_invalid_file(self):

        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # When, Then
        with self.assertRaises(ValueError):
            visualization.convert_img_slices(temp_file) 

        os.remove(temp_file)

    def test_display_single_slice(self):

        # Given
        label = QLabel()
        label.setFixedSize(400, 300)

        data = np.random.rand(300, 400)
        data_uint8 = (data * 255).astype(np.uint8)
        qimage = QImage(data_uint8.data, data_uint8.shape[1], data_uint8.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # When
        visualization.display_single_slice(label, pixmap)

        # Then
        self.assertIsNotNone(label.pixmap())

class TestSegmentation(unittest.TestCase):
    def test_preprocess_img(self):

        # Given
        data = np.random.rand(128, 128, 128)

        # When
        img_tensor = segmentation.preprocess_img(data)

        # Then
        self.assertEqual(img_tensor.shape, (1, 128, 128, 128))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))

    def test_segment_img(self):

        # Given
        model_instance = model.get_unet_model(num_classes=1, in_channels=1)
        img_tensor = torch.rand((1, 128, 128, 128))

        # When
        seg_pred = segmentation.segment_img(model_instance, img_tensor)

        # Then
        self.assertEqual(seg_pred.shape, (128, 128, 128))
        self.assertTrue(isinstance(seg_pred, np.ndarray))

    def test_save_segmentation(self):

        # Given
        seg_out = np.random.randint(0, 2, (128, 128, 128))
        affine = np.eye(4)
        save_path = 'temp_segmentation.nii.gz'

        # When
        segmentation.save_segmentation(seg_out, affine, save_path)

        # Then
        self.assertTrue(os.path.exists(save_path))
        loaded_seg = nib.load(save_path).get_fdata()
        np.testing.assert_array_equal(seg_out, loaded_seg)

        os.remove(save_path)

class TestModel(unittest.TestCase):
    def test_get_unet_model(self):

        # Given
        num_classes = 1
        in_channels = 1

        # When
        net = model.get_unet_model(num_classes=num_classes, in_channels=in_channels)

        # Then
        self.assertIsNotNone(net)
        self.assertEqual(net.in_channels, in_channels)
        self.assertEqual(net.out_channels, num_classes)
        from monai.networks.nets import UNet
        self.assertIsInstance(net, UNet)

    def test_get_unet_model_invalid_params(self):
        # Given, When, Then
        with self.assertRaises(Exception):
            model.get_unet_model(num_classes=-1, in_channels=1) # since the function doesn't have input validation, error must come within UNet


    def test_dice_metric(self):
        # Given
        from monai.metrics import DiceMetric
        # When
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        pred = torch.rand(1, 2, 10, 10, 10)
        label = torch.randint(0, 2, (1, 2, 10, 10, 10))
        dice = dice_metric(pred, label)
        # Then
        self.assertTrue(0 <= dice.item() <= 1)

class TestDemo(unittest.TestCase):
    def test_convert_to_stl(self):

        # Given
        data = np.zeros((10, 10, 10))
        data[3:7, 3:7, 3:7] = 1  # 3x3x3 cube
        out_path = 'temp_mesh.stl'

        # When
        demo.convert_to_stl(data, out_path)

        # Then
        self.assertTrue(os.path.exists(out_path))
        from stl import mesh as stl_mesh
        obj_3d = stl_mesh.Mesh.from_file(out_path)
        self.assertIsNotNone(obj_3d)

        os.remove(out_path)

    def test_convert_to_stl_empty_data(self):

        # Given
        data = np.zeros((10, 10, 10))
        out_path = 'temp_mesh_empty.stl'

        # When
        demo.convert_to_stl(data, out_path)

        # Then
        self.assertFalse(os.path.exists(out_path))

class TestMedicalImageViewer(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.viewer = MedicalImageViewer()

    def test_log_message(self):

        # Given
        self.viewer.error_log = MagicMock()

        # When 
        self.viewer.log_message("Test message")

        # Then
        self.viewer.error_log.appendPlainText.assert_called_with("Test message")

    def test_get_distinct_colors(self):

        # Given
        N = 5

        # When
        colors = self.viewer.get_distinct_colors(N)

        # Then
        self.assertEqual(len(colors), N)

    def test_init_ui(self):
        # Given, When
        try:
            self.viewer.init_ui()
        except Exception as e:
            # Then
            self.fail(f"init_ui raised an exception: {e}")

    def test_setup_menu_bar(self):

        # Given, When
        self.viewer.setup_menu_bar()

        # Then
        menu_bar = self.viewer.menuBar()
        file_menu = menu_bar.actions()[0].menu()
        actions = [action.text() for action in file_menu.actions()]
        self.assertIn('Upload data', actions)
        self.assertIn('Save segmentation', actions)
        self.assertIn('Close segmentation', actions)

    def test_on_upload_data_cancelled(self):
        # Given
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('', False)):
            # When
            self.viewer.on_upload_data()
            # Then
            with patch.object(self.viewer, 'on_upload_ct_scan') as mock_upload_ct, \
                 patch.object(self.viewer, 'on_upload_segmented_ct_scan') as mock_upload_seg:
                self.viewer.on_upload_data()
                mock_upload_ct.assert_not_called()
                mock_upload_seg.assert_not_called()

    def test_on_upload_data_upload_ct_scan(self):

        # Given
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=("Upload CT scan for both segmentation and visualization.", True)), \
             patch.object(self.viewer, 'on_upload_ct_scan') as mock_on_upload_ct_scan:

            # When
            self.viewer.on_upload_data()

            # Then
            mock_on_upload_ct_scan.assert_called_once()

    def test_on_upload_ct_scan_valid_file(self):

        # Given
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=['valid.nii']), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((10,10,10)), np.eye(4))), \
             patch('visualization.convert_img_slices', return_value=([], [], [], np.eye(4))), \
             patch.object(self.viewer, 'update_image_placeholders'), \
             patch.object(self.viewer, 'render_3d_visualization'):

            # When
            self.viewer.on_upload_ct_scan()
            self.assertIsNotNone(self.viewer.ct_scans)

            # Then
            self.assertTrue(self.viewer.segment_action.isEnabled())

    def test_on_upload_ct_scan_invalid_file(self):

        # Given
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=['invalid.nii']), \
             patch('data_import.validate_nifti', return_value=False), \
             patch.object(QMessageBox, 'critical') as mock_critical:

             # When
            self.viewer.on_upload_ct_scan()

            # Then
            mock_critical.assert_called()

    def test_on_upload_segmented_ct_scan(self):

        # Given
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=['segmented.nii']), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((10,10,10)), np.eye(4))), \
             patch.object(self.viewer, 'render_3d_visualization_from_data'):

            # When
            self.viewer.on_upload_segmented_ct_scan()

            # Then
            self.assertIsNotNone(self.viewer.segmentation_result)
            self.viewer.render_3d_visualization_from_data.assert_called()

    def test_on_segment_image_no_ct_scans(self):

        # Given
        self.viewer.ct_scans = None

        # When
        with patch.object(QMessageBox, 'warning') as mock_warning:
            self.viewer.on_segment_image()

            # Then
            mock_warning.assert_called()
            args = mock_warning.call_args[0]
            self.assertIn("Please upload your CT scan", args[2])

    def test_on_segment_image_with_ct_scans(self):

        # Given
        self.viewer.ct_scans = np.zeros((128, 128, 128))

        # When
        with patch.object(self.viewer, 'load_segmentation_model', return_value=MagicMock()), \
             patch('segmentation.preprocess_img', return_value=torch.zeros((1, 128, 128, 128))), \
             patch('segmentation.segment_img', return_value=np.zeros((128, 128, 128))), \
             patch.object(self.viewer, 'render_3d_visualization_from_data'):

            self.viewer.on_segment_image()

            # Then
            self.assertIsNotNone(self.viewer.segmentation_result)

    def test_load_segmentation_model(self):

        # Given
        with patch('GUI.torch.load', return_value={}), \
             patch('GUI.get_unet_model', return_value=MagicMock()):

            # When
            net = self.viewer.load_segmentation_model()

            # Then
            self.assertIsNotNone(net)

    def test_load_segmentation_model_file_not_found(self):

        # Given 
        with patch('GUI.get_unet_model', return_value=MagicMock()), \
             patch('GUI.torch.load', side_effect=FileNotFoundError), \
             patch.object(QMessageBox, 'critical') as mock_critical, \
             patch.object(self.viewer, 'log_message') as mock_log_message:
             
            # When 
            with self.assertRaises(FileNotFoundError):
                self.viewer.load_segmentation_model()

            # Then 
            mock_critical.assert_called()
            mock_log_message.assert_called_with("ERROR: best_metric_model.pth not found.")

    def test_update_image_placeholders(self):

        # Given
        self.viewer.scan_list_sagittal = [QPixmap(100, 100) for _ in range(1)]
        self.viewer.scan_list_coronal = [QPixmap(100, 100) for _ in range(1)]
        self.viewer.scan_list_axial = [QPixmap(100, 100) for _ in range(1)]
        self.viewer.scan_top_left = QLabel()
        self.viewer.scan_top_right = QLabel()
        self.viewer.scan_bottom_left = QLabel()

        # When
        self.viewer.update_image_placeholders()

        # Then
        self.assertIsNotNone(self.viewer.scan_top_left.pixmap())
        self.assertIsNotNone(self.viewer.scan_top_right.pixmap())
        self.assertIsNotNone(self.viewer.scan_bottom_left.pixmap())

    def test_on_save_segmentation_with_data(self):

        # Given
        self.viewer.segmentation_result = np.zeros((10, 10, 10))

        # When
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=('segmentation_result.nii.gz', '')), \
             patch('segmentation.save_segmentation') as mock_save, \
             patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_save_segmentation()

            # Then
            mock_save.assert_called()
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("Segmentation has been saved at", args[2])

    def test_on_save_segmentation_no_data(self):

        # Given
        self.viewer.segmentation_result = None

        # When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_save_segmentation()

            # Then
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("There is no segmentation data to save.", args[2])

    def test_on_close_segmentation(self):

        # Given
        self.viewer.ct_scans = np.zeros((10,10,10))
        self.viewer.segmentation_result = np.zeros((10,10,10))

        # When
        self.viewer.on_close_segmentation()

        # Then
        self.assertIsNone(self.viewer.ct_scans)
        self.assertIsNone(self.viewer.segmentation_result)

    def test_on_zoom_in(self):
        # Given
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        self.viewer.on_zoom_in()

        # Then
        self.viewer.plotter.zoom.assert_called_with(1.2)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_on_zoom_out(self):

        # Given
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        self.viewer.on_zoom_out()

        # Then
        self.viewer.plotter.zoom.assert_called_with(0.8)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_on_help(self):

        # Given, When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_help()

            # Then
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("SegMed help information:", args[2])

    def test_on_report_problem(self):

        # Given, When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_report_problem()

            # Then
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("Please provide us with the details of the problem you encountered.", args[2])

    def test_on_about(self):

        # Given, When
        with patch.object(QDialog, 'exec_') as mock_exec:
            self.viewer.on_about()

            # Then
            mock_exec.assert_called()

    def test_on_slider_move(self):

        # Given
        self.viewer.scan_list_sagittal = [QPixmap(100, 100) for _ in range(10)]
        self.viewer.scan_list_coronal = [QPixmap(100, 100) for _ in range(10)]
        self.viewer.scan_list_axial = [QPixmap(100, 100) for _ in range(10)]
        self.viewer.scan_top_left = QLabel()
        self.viewer.scan_top_right = QLabel()
        self.viewer.scan_bottom_left = QLabel()
        self.viewer.slider_sagittal.setMaximum(9)
        self.viewer.slider_coronal.setMaximum(9)
        self.viewer.slider_axial.setMaximum(9)
        self.viewer.slider_sagittal.setValue(5)
        self.viewer.slider_coronal.setValue(5)
        self.viewer.slider_axial.setValue(5)

        # When
        try:
            self.viewer.on_slider_move()
        except Exception as e:

            # Then
            self.fail(f"on_slider_move raised an exception: {e}")

    def test_render_3d_visualization_from_data(self):

        # Given
        seg_data = np.zeros((50, 50, 50))
        seg_data[20:30, 20:30, 20:30] = 1  # cube to ensure valid STL

        # When
        with patch.object(self.viewer, 'vtk_widget', MagicMock()), \
             patch('GUI.Plotter') as mock_plotter_class, \
             patch('GUI.load', return_value=MagicMock()), \
             patch('demo.convert_to_stl'):

            mock_plotter_instance = mock_plotter_class.return_value
            self.viewer.render_3d_visualization_from_data(seg_data)

            # Then
            mock_plotter_instance.show.assert_called()
            self.viewer.vtk_widget.update.assert_called()

class TestPerformance(unittest.TestCase):
    def test_full_pipeline(self):
        # Given
        ct_scan = np.random.rand(128, 128, 128)
        affine = np.eye(4)
        nifti_path = "temp_ct.nii.gz"
        nib.save(nib.Nifti1Image(ct_scan, affine), nifti_path)

        # When
        processed = segmentation.preprocess_img(ct_scan)
        model_instance = model.get_unet_model(num_classes=2, in_channels=1)
        output = segmentation.segment_img(model_instance, processed)
        segmentation.save_segmentation(output, affine, "temp_output.nii.gz")

        # Then
        self.assertTrue(os.path.exists("temp_output.nii.gz"))
        os.remove(nifti_path)
        os.remove("temp_output.nii.gz")


    def test_segmentation_speed(self):
        # Given
        ct_scan = np.random.rand(128, 128, 128).astype(np.float32)

        # When
        start_preprocess = time.time()
        processed_scan = preprocess_img(ct_scan, target_size=(128, 128, 128))
        end_preprocess = time.time()
        preprocess_time = end_preprocess - start_preprocess
        print(f"Preprocessing time: {preprocess_time:.4f} seconds")

        model = get_unet_model(num_classes=2, in_channels=1)
        model.eval()
        start_segmentation = time.time()
        segmentation_result = segment_img(model, processed_scan)
        end_segmentation = time.time()
        segmentation_time = end_segmentation - start_segmentation
        print(f"Segmentation time: {segmentation_time:.4f} seconds")

        # Then
        assert preprocess_time < 5.0, "Preprocessing time exceeded 5 seconds"
        assert segmentation_time < 10.0, "Segmentation time exceeded 10 seconds"


def test_visualization_speed():
    # Given
    ct_slice = np.random.randint(0, 255, size=(128, 128)).astype(np.uint8)
    pixmap = QPixmap(128, 128)
    label = QLabel()
    label.setFixedSize(400, 300)

    # When
    start_visualization = time.time()
    display_single_slice(label, pixmap)
    end_visualization = time.time()
    visualization_time = end_visualization - start_visualization
    print(f"Visualization time: {visualization_time:.4f} seconds")

    # Then
    assert visualization_time < 4.0, "Visualization time exceeded 4 second"
    

if __name__ == '__main__':
    unittest.main()
