# === test_application.py ===

import unittest
import os
import numpy as np
import nibabel as nib
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
from unittest.mock import patch, MagicMock

# Import the modules to be tested
import data_import
import visualization
import segmentation
import model
import demo

# Import the MedicalImageViewer class from GUI.py
from GUI import MedicalImageViewer

class TestDataImport(unittest.TestCase):
    def test_validate_nifti_valid(self):

        data = np.zeros((10, 10, 10))
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_valid_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # Test validate_nifti
        self.assertTrue(data_import.validate_nifti(temp_file))

        os.remove(temp_file)

    def test_validate_nifti_invalid(self):

        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # Test validate_nifti
        self.assertFalse(data_import.validate_nifti(temp_file))

        os.remove(temp_file)

    def test_load_nifti_valid(self):

        data = np.random.rand(10, 10, 10)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_valid_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # Test load_nifti
        loaded_data, loaded_affine = data_import.load_nifti(temp_file)
        np.testing.assert_array_equal(data, loaded_data)
        np.testing.assert_array_equal(affine, loaded_affine)

        os.remove(temp_file)

    def test_load_nifti_invalid(self):

        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # Test load_nifti, should raise ValueError
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

        data = np.random.rand(50, 50, 50)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # Test convert_img_slices
        scan_list_sagittal, scan_list_coronal, scan_list_axial, loaded_affine = visualization.convert_img_slices(temp_file)

        self.assertEqual(len(scan_list_sagittal), 128)
        self.assertEqual(len(scan_list_coronal), 300)
        self.assertEqual(len(scan_list_axial), 400)

        os.remove(temp_file)

    def test_convert_img_slices_invalid_file(self):
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('This is not a NIfTI file')

        # Should raise ValueError
        with self.assertRaises(ValueError):
            visualization.convert_img_slices(temp_file)

        os.remove(temp_file)

    def test_display_single_slice(self):
        label = QLabel()
        label.setFixedSize(400, 300)

        data = np.random.rand(300, 400)
        data_uint8 = (data * 255).astype(np.uint8)
        qimage = QImage(data_uint8.data, data_uint8.shape[1], data_uint8.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # Test display_single_slice
        visualization.display_single_slice(label, pixmap)
        self.assertIsNotNone(label.pixmap())

class TestSegmentation(unittest.TestCase):
    def test_preprocess_img(self):
        data = np.random.rand(128, 128, 128)

        # Call preprocess_img
        img_tensor = segmentation.preprocess_img(data)
        self.assertEqual(img_tensor.shape, (1, 128, 128, 128))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))

    def test_segment_img(self):
        # Create dummy model
        model_instance = model.get_unet_model(num_classes=1, in_channels=1)

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

        loaded_seg = nib.load(save_path).get_fdata()
        np.testing.assert_array_equal(seg_out, loaded_seg)

        os.remove(save_path)

class TestModel(unittest.TestCase):
    def test_get_unet_model(self):
        num_classes = 1
        in_channels = 1
        net = model.get_unet_model(num_classes=num_classes, in_channels=in_channels)
        self.assertIsNotNone(net)
        self.assertEqual(net.in_channels, in_channels)
        self.assertEqual(net.out_channels, num_classes)
        # Check that net is an instance of UNet
        from monai.networks.nets import UNet
        self.assertIsInstance(net, UNet)

    def test_get_unet_model_invalid_params(self):
        # Since the function doesn't have input validation, we expect an error from within UNet
        with self.assertRaises(Exception):
            model.get_unet_model(num_classes=-1, in_channels=1)

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

        from stl import mesh as stl_mesh
        obj_3d = stl_mesh.Mesh.from_file(out_path)
        self.assertIsNotNone(obj_3d)

        os.remove(out_path)

    def test_convert_to_stl_empty_data(self):
        # Create an empty numpy array
        data = np.zeros((10, 10, 10))
        out_path = 'temp_mesh_empty.stl'

        demo.convert_to_stl(data, out_path)

        # Check that file does not exist (since the data is empty)
        self.assertFalse(os.path.exists(out_path))

class TestMedicalImageViewer(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.viewer = MedicalImageViewer()

    def test_log_message(self):
        # Test that log_message appends text to error_log
        self.viewer.error_log = MagicMock()
        self.viewer.log_message("Test message")
        self.viewer.error_log.appendPlainText.assert_called_with("Test message")

    def test_get_distinct_colors(self):
        # Test that get_distinct_colors returns N colors
        N = 5
        colors = self.viewer.get_distinct_colors(N)
        self.assertEqual(len(colors), N)

    def test_init_ui(self):
        # Test that init_ui runs without errors
        try:
            self.viewer.init_ui()
        except Exception as e:
            self.fail(f"init_ui raised an exception: {e}")

    def test_setup_menu_bar(self):
        # Test that menu bar is set up correctly
        self.viewer.setup_menu_bar()
        menu_bar = self.viewer.menuBar()
        file_menu = menu_bar.actions()[0].menu()
        actions = [action.text() for action in file_menu.actions()]
        self.assertIn('Upload data', actions)
        self.assertIn('Save segmentation', actions)
        self.assertIn('Close segmentation', actions)

    def test_on_upload_data_cancelled(self):
        # Test on_upload_data when user cancels the dialog
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('', False)):
            self.viewer.on_upload_data()
            # Since the user cancelled, no action should be taken

    def test_on_upload_data_upload_ct_scan(self):
        # Test on_upload_data when user chooses to upload CT scan
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=("Upload CT scan for both segmentation and visualization.", True)), \
             patch.object(self.viewer, 'on_upload_ct_scan') as mock_on_upload_ct_scan:
            self.viewer.on_upload_data()
            mock_on_upload_ct_scan.assert_called_once()

    def test_on_upload_ct_scan_valid_file(self):
        # Test uploading a valid CT scan
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=['valid.nii']), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((10,10,10)), np.eye(4))), \
             patch('visualization.convert_img_slices', return_value=([], [], [], np.eye(4))), \
             patch.object(self.viewer, 'update_image_placeholders'), \
             patch.object(self.viewer, 'render_3d_visualization'):
            self.viewer.on_upload_ct_scan()
            # Check that ct_scans is set
            self.assertIsNotNone(self.viewer.ct_scans)
            # Check that segment_action is enabled
            self.assertTrue(self.viewer.segment_action.isEnabled())

    def test_on_upload_ct_scan_invalid_file(self):
        # Test uploading an invalid CT scan
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=['invalid.nii']), \
             patch('data_import.validate_nifti', return_value=False), \
             patch.object(QMessageBox, 'critical') as mock_critical:
            self.viewer.on_upload_ct_scan()
            mock_critical.assert_called()

    def test_on_upload_segmented_ct_scan(self):
        # Test uploading an already segmented CT scan
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=['segmented.nii']), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((10,10,10)), np.eye(4))), \
             patch.object(self.viewer, 'render_3d_visualization_from_data'):
            self.viewer.on_upload_segmented_ct_scan()
            self.assertIsNotNone(self.viewer.segmentation_result)
            self.viewer.render_3d_visualization_from_data.assert_called()

    def test_on_segment_image_no_ct_scans(self):
        # test when no ct scans is loaded
        self.viewer.ct_scans = None
        with patch.object(QMessageBox, 'warning') as mock_warning:
            self.viewer.on_segment_image()
            mock_warning.assert_called()
            args = mock_warning.call_args[0]
            self.assertIn("Please upload your CT scan", args[2])

    def test_on_segment_image_with_ct_scans(self):
        # Test on_segment_image with CT scans loaded
        self.viewer.ct_scans = np.zeros((128, 128, 128))
        with patch.object(self.viewer, 'load_segmentation_model', return_value=MagicMock()), \
             patch('segmentation.preprocess_img', return_value=torch.zeros((1, 128, 128, 128))), \
             patch('segmentation.segment_img', return_value=np.zeros((128, 128, 128))), \
             patch.object(self.viewer, 'render_3d_visualization_from_data'):
            self.viewer.on_segment_image()
            self.assertIsNotNone(self.viewer.segmentation_result)

    def test_load_segmentation_model(self):
        # tests loading the segmentation model
        with patch('GUI.torch.load', return_value={}), \
             patch('GUI.get_unet_model', return_value=MagicMock()):
            model = self.viewer.load_segmentation_model()
            self.assertIsNotNone(model)

    def test_load_segmentation_model_file_not_found(self):
        # Test error handling when model file is not found
        with patch('GUI.get_unet_model', return_value=MagicMock()), \
                patch('GUI.torch.load', side_effect=FileNotFoundError), \
                patch.object(QMessageBox, 'critical') as mock_critical:
            # We'll also patch the log_message method to capture the log
            with patch.object(self.viewer, 'log_message') as mock_log_message:
                with self.assertRaises(FileNotFoundError):
                    self.viewer.load_segmentation_model()
                mock_critical.assert_called()
                # Check that the error message was logged
                mock_log_message.assert_called_with("ERROR: best_metric_model.pth not found.")

    def test_update_image_placeholders(self):
        # Test updating image placeholders
        self.viewer.scan_list_sagittal = [QPixmap(100, 100) for _ in range(1)]
        self.viewer.scan_list_coronal = [QPixmap(100, 100) for _ in range(1)]
        self.viewer.scan_list_axial = [QPixmap(100, 100) for _ in range(1)]
        self.viewer.scan_top_left = QLabel()
        self.viewer.scan_top_right = QLabel()
        self.viewer.scan_bottom_left = QLabel()
        self.viewer.update_image_placeholders()
        self.assertIsNotNone(self.viewer.scan_top_left.pixmap())
        self.assertIsNotNone(self.viewer.scan_top_right.pixmap())
        self.assertIsNotNone(self.viewer.scan_bottom_left.pixmap())

    def test_on_save_segmentation_with_data(self):
        # test saving segmentation when data is available
        self.viewer.segmentation_result = np.zeros((10, 10, 10))
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=('segmentation_result.nii.gz', '')), \
             patch('segmentation.save_segmentation') as mock_save, \
             patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_save_segmentation()
            mock_save.assert_called()
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("Segmentation has been saved at", args[2])

    def test_on_save_segmentation_no_data(self):
        # Test saving segmentation when no data is available
        self.viewer.segmentation_result = None
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_save_segmentation()
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("There is no segmentation data to save.", args[2])

    def test_on_close_segmentation(self):
        # test closing the segmentation
        self.viewer.ct_scans = np.zeros((10,10,10))
        self.viewer.segmentation_result = np.zeros((10,10,10))
        self.viewer.on_close_segmentation()
        self.assertIsNone(self.viewer.ct_scans)
        self.assertIsNone(self.viewer.segmentation_result)

    def test_on_zoom_in(self):
        # Test zoom in functionality
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()
        self.viewer.on_zoom_in()
        self.viewer.plotter.zoom.assert_called_with(1.2)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_on_zoom_out(self):
        # Test zoom out functionality
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()
        self.viewer.on_zoom_out()
        self.viewer.plotter.zoom.assert_called_with(0.8)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_on_help(self):
        # testing the help dialog display
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_help()
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("SegMed help information", args[2])

    def test_on_report_problem(self):
        # testing the report problem dialog display
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.on_report_problem()
            mock_info.assert_called()
            args = mock_info.call_args[0]
            self.assertIn("Please provide us with the details of the problem", args[2])

    def test_on_about(self):
        # Test about dialog
        with patch.object(QDialog, 'exec_') as mock_exec:
            self.viewer.on_about()
            mock_exec.assert_called()

    def test_on_slider_move(self):
        # test the slider movement is working
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
        try:
            self.viewer.on_slider_move()
        except Exception as e:
            self.fail(f"on_slider_move raised an exception: {e}")

    def test_render_3d_visualization_from_data(self):
        # test rendering 3D visualization from data
        seg_data = np.zeros((50, 50, 50))
        seg_data[20:30, 20:30, 20:30] = 1  # Create a cube to ensure valid STL
        with patch.object(self.viewer, 'vtk_widget', MagicMock()), \
             patch('GUI.Plotter') as mock_plotter_class, \
             patch('GUI.load', return_value=MagicMock()), \
             patch('demo.convert_to_stl'):
            mock_plotter_instance = mock_plotter_class.return_value
            self.viewer.render_3d_visualization_from_data(seg_data)
            mock_plotter_instance.show.assert_called()
            self.viewer.vtk_widget.update.assert_called()

    def test_render_3d_visualization(self):
        # testing rendering when no segmentation file provided
        with patch.object(self.viewer, 'log_message') as mock_log:
            self.viewer.render_3d_visualization()
            mock_log.assert_any_call("No segmentation file provided, visualization skipped.")

if __name__ == '__main__':
    unittest.main()
