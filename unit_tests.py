# === unit_tests.py ===
import unittest
import os
import sys
import torch
import time
import numpy as np
import nibabel as nib

from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox, QDialog, QInputDialog
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

from unittest.mock import patch, MagicMock


################################################################################
#                      T E S T   C L A S S E S
################################################################################

class TestDataImport(unittest.TestCase):
    """ Tests for data_import.py """

    def test_validate_nifti_valid(self):
        """
        Checks whether function 'validate_nifti' correctly identifies a valid NIfTI file.
        """
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

        # Cleanup
        os.remove(temp_file)

    def test_validate_nifti_invalid(self):
        """
        Checks whether function 'validate_nifti' correctly identifies an invalid file (like a .txt).
        """
        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('Not a NIfTI file.')

        # When
        result = data_import.validate_nifti(temp_file)

        # Then
        self.assertFalse(result)

        # Cleanup
        os.remove(temp_file)

    def test_load_nifti_valid(self):
        """
        Checks whether function 'load_nifti' correctly loads data and affine from a NIfTI file.
        """
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

        # Cleanup
        os.remove(temp_file)

    def test_load_nifti_invalid(self):
        """
        Checks the response of 'load_nifti' to invalid NIfTI files.
        """
        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('Invalid NIfTI file.')

        # When & Then
        with self.assertRaises(ValueError):
            data_import.load_nifti(temp_file)

        # Cleanup
        os.remove(temp_file)


class TestVisualization(unittest.TestCase):
    """ Tests for visualization.py """

    def setUp(self):
        # Given: PyQt must have a running QApplication instance.
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

    def test_convert_img_slices(self):
        """
        Checks whether 'convert_img_slices' correctly splits a NIfTI file into
        scans in sagittal, coronal, axial axes (planes).
        """
        # Given
        data = np.random.rand(50, 50, 50).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # When
        # Using the defaults: target_size=(400,300,128)
        scan_list_sagittal, scan_list_coronal, scan_list_axial, loaded_affine = visualization.convert_img_slices(
            temp_file
        )

        # Then
        # By default, out_array_sagittal -> length=128, coronal=300, axial=400
        self.assertEqual(len(scan_list_sagittal), 128)
        self.assertEqual(len(scan_list_coronal), 300)
        self.assertEqual(len(scan_list_axial), 400)

        # Cleanup
        os.remove(temp_file)

    def test_convert_img_slices_invalid_file(self):
        """
        Checks the response of 'convert_img_slices' to invalid NIfTI files.
        """
        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('Invalid NIfTI file.')

        # When & Then
        with self.assertRaises(ValueError):
            visualization.convert_img_slices(temp_file)

        # Cleanup
        os.remove(temp_file)

    def test_display_single_slice(self):
        """
        Checks whether 'display_single_slice' correctly displays a single slice
        of a NIfTI file in a QLabel widget.
        """
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
    """ Tests for segmentation.py """

    def test_preprocess_img(self):
        """
        Checks whether 'preprocess_img' correctly preprocesses a 3D numpy array into a torch tensor.
        """
        # Given
        data = np.random.rand(128, 128, 128).astype(np.float32)

        # When
        img_tensor = segmentation.preprocess_img(data)

        # Then
        self.assertEqual(img_tensor.shape, (1, 128, 128, 128))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))

    def test_segment_img(self):
        """
        Checks whether 'segment_img' correctly segments a 3D numpy array using a trained U-Net model.
        """
        # Given
        model_instance = get_unet_model(num_classes=1, in_channels=1)
        img_tensor = torch.rand((1, 128, 128, 128))

        # When
        seg_pred = segmentation.segment_img(model_instance, img_tensor)

        # Then
        self.assertEqual(seg_pred.shape, (128, 128, 128))
        self.assertTrue(isinstance(seg_pred, np.ndarray))

    def test_save_segmentation(self):
        """
        Checks whether 'save_segmentation' correctly saves a 3D numpy array as a NIfTI file.
        """
        # Given
        seg_out = np.random.randint(0, 2, (128, 128, 128)).astype(np.int16)
        affine = np.eye(4)
        save_path = 'temp_segmentation.nii.gz'

        # When
        segmentation.save_segmentation(seg_out, affine, save_path)

        # Then
        self.assertTrue(os.path.exists(save_path))

        loaded_seg = nib.load(save_path).get_fdata()
        np.testing.assert_array_equal(seg_out, loaded_seg)

        # Cleanup
        os.remove(save_path)


class TestModel(unittest.TestCase):
    """ Tests for model.py """

    def test_get_unet_model(self):
        """
        Checks whether 'get_unet_model' correctly returns a U-Net model with specified parameters.
        """
        # Given
        num_classes = 1
        in_channels = 1

        # When
        net = get_unet_model(num_classes=num_classes, in_channels=in_channels)

        # Then
        self.assertIsNotNone(net)
        self.assertEqual(net.in_channels, in_channels)
        self.assertEqual(net.out_channels, num_classes)

    def test_get_unet_model_invalid_params(self):
        """
        Checks whether 'get_unet_model' raises an exception for invalid parameters.
        """
        # Given
        invalid_num_classes = -1
        in_channels = 1

        # When & Then
        with self.assertRaises(Exception):
            get_unet_model(num_classes=invalid_num_classes, in_channels=in_channels)  # invalid num_classes


class TestDemo(unittest.TestCase):
    """ Tests for demo.py """

    def test_convert_to_stl(self):
        """
        Checks whether 'convert_to_stl' correctly converts a 3D numpy array into a STL file.
        """
        # Given
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[3:7, 3:7, 3:7] = 1  # 3x3x3 cube
        out_path = 'temp_mesh.stl'

        # When
        demo.convert_to_stl(data, out_path)

        # Then
        self.assertTrue(os.path.exists(out_path))

        from stl import mesh as stl_mesh
        obj_3d = stl_mesh.Mesh.from_file(out_path)
        self.assertIsNotNone(obj_3d)

        # Cleanup
        os.remove(out_path)

    def test_convert_to_stl_empty_data(self):
        """
        Checks whether 'convert_to_stl' returns without creating an STL file if the data is empty.
        """
        # Given
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        out_path = 'temp_mesh_empty.stl'

        # When
        demo.convert_to_stl(data, out_path)

        # Then
        # No file created if data is all zeros.
        self.assertFalse(os.path.exists(out_path))


class TestGUI(unittest.TestCase):
    """
    Tests for GUI.py (MedicalImageViewer class).
    Many of these tests rely heavily on PyQt5 and mock objects.
    """

    def setUp(self):
        # Given
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.viewer = MedicalImageViewer()

    def test_log_message(self):
        """
        Checks whether logs error messages in the GUI widget are visible.
        """
        # Given
        self.viewer.error_log = MagicMock()

        # When
        self.viewer.log_message("Test message")

        # Then
        self.viewer.error_log.appendPlainText.assert_called_with("Test message")

    def test_init_ui(self):
        """
        Checks whether function 'init_ui' correctly sets up the user interface.
        """
        # Given & When & Then
        try:
            self.viewer.init_ui()
        except Exception as e:
            self.fail(f"init_ui raised an exception: {e}")

    def test_setup_menu_bar(self):
        """
        Checks whether function 'setup_menu_bar' correctly sets up the menu bar with expected actions.
        """
        # Given
        self.viewer.setup_menu_bar()

        # When
        menu_bar = self.viewer.menuBar()
        file_menu = menu_bar.actions()[0].menu()
        actions = [action.text() for action in file_menu.actions()]

        # Then
        self.assertIn('Upload data', actions)
        self.assertIn('Save segmentation', actions)
        self.assertIn('Close segmentation', actions)

    def test_on_upload_segmented_ct_scan(self):
        """
        Checks whether function 'on_upload_segmented_ct_scan' correctly handles the upload of valid segmentation.
        """
        # Given
        file_selection = ['segmented.nii']

        # When
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
                patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=file_selection), \
                patch('data_import.validate_nifti', return_value=True), \
                patch('data_import.load_nifti', return_value=(np.zeros((10, 10, 10)), np.eye(4))), \
                patch.object(self.viewer, 'render_3d_visualization_from_data', MagicMock()) as mock_render:
            self.viewer.on_upload_segmented_ct_scan()

        # Then
        self.assertIsNotNone(self.viewer.segmentation_result)
        mock_render.assert_called_once()

    def test_on_upload_data_cancelled(self):
        """
        Checks whether function 'on_upload_data' exits gracefully if user cancels.
        """
        # Given
        user_selection = ('', False)

        # When
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=user_selection):
            with patch.object(self.viewer, 'on_upload_ct_scan') as mock_upload_ct, \
                 patch.object(self.viewer, 'on_upload_segmented_ct_scan') as mock_upload_seg:
                self.viewer.on_upload_data()

        # Then
        mock_upload_ct.assert_not_called()
        mock_upload_seg.assert_not_called()

    def test_on_upload_ct_scan_valid_file(self):
        """
        Checks whether function 'on_upload_ct_scan' correctly handles a valid NIfTI file.
        """
        # Given
        file_selection = ['valid.nii']

        # When
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=file_selection), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((10, 10, 10)), np.eye(4))), \
             patch('visualization.convert_img_slices', return_value=([], [], [], np.eye(4))), \
             patch.object(self.viewer, 'update_image_placeholders'), \
             patch.object(self.viewer, 'render_3d_visualization'):
            self.viewer.on_upload_ct_scan()

        # Then
        self.assertIsNotNone(self.viewer.ct_scans)
        self.assertTrue(self.viewer.segment_action.isEnabled())

    def test_on_upload_ct_scan_invalid_file(self):
        """
        Checks whether function 'on_upload_ct_scan' correctly handles an invalid CT scan file.
        """
        # Given
        file_selection = ['invalid.nii']

        # When
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=file_selection), \
             patch('data_import.validate_nifti', return_value=False), \
             patch.object(QMessageBox, 'critical') as mock_critical:
            self.viewer.on_upload_ct_scan()

        # Then
        mock_critical.assert_called()


    def test_on_segment_image_no_ct_scans(self):
        """
        Checks whether function 'on_segment_image' warns if no CT scans were uploaded.
        """
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
        """
        Checks whether function 'on_segment_image' proceeds if CT scans are loaded.
        """
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
        """
        Checks whether function 'load_segmentation_model' correctly loads the segmentation model.
        """
        # Given
        with patch('GUI.torch.load', return_value={}), \
             patch('GUI.get_unet_model', return_value=MagicMock()):

            # When
            net = self.viewer.load_segmentation_model()

            # Then
            self.assertIsNotNone(net)

    def test_load_segmentation_model_file_not_found(self):
        """
        Checks whether function 'load_segmentation_model' handles a missing model file.
        """
        # Given
        with patch('GUI.get_unet_model', return_value=MagicMock()), \
             patch('GUI.torch.load', side_effect=FileNotFoundError), \
             patch.object(QMessageBox, 'critical') as mock_critical, \
             patch.object(self.viewer, 'log_message') as mock_log_message:

            # When & Then
            with self.assertRaises(FileNotFoundError):
                self.viewer.load_segmentation_model()

            mock_critical.assert_called()
            mock_log_message.assert_called_with("ERROR: best_metric_model3.pth not found.")

    def test_update_image_placeholders(self):
        """
        Checks whether function 'update_image_placeholders' updates all three placeholders.
        """
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
        """
        Checks whether function 'on_save_segmentation' handles a valid segmentation save operation.
        """
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
        """
        Checks whether function 'on_save_segmentation' warns if no segmentation data is available.
        """
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
        """
        Checks whether function 'on_close_segmentation' resets the viewer to initial state.
        """
        # Given
        self.viewer.ct_scans = np.zeros((10, 10, 10))
        self.viewer.segmentation_result = np.zeros((10, 10, 10))

        # When
        self.viewer.on_close_segmentation()

        # Then
        self.assertIsNone(self.viewer.ct_scans)
        self.assertIsNone(self.viewer.segmentation_result)

    def test_on_zoom_in(self):
        """
        Checks whether function 'on_zoom_in' triggers the corresponding zoom logic.
        """
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
        """
        Checks whether function 'on_zoom_out' triggers the corresponding zoom logic.
        """
        # Given
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        self.viewer.on_zoom_out()

        # Then
        self.viewer.plotter.zoom.assert_called_with(0.8)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()



    def test_on_report_problem(self):
        """
        Checks whether function 'on_report_problem' opens the default mail client.
        """
        # Given
        # (No specific setup required)

        # When
        with patch('PyQt5.QtGui.QDesktopServices.openUrl') as mock_open_url:
            self.viewer.on_report_problem()

        # Then
        # Assert that openUrl was called with the correct QUrl
        mock_open_url.assert_called_once()

    def test_on_about(self):
        """
        Checks whether function 'on_about' opens the about dialog without error.
        """
        # Given
        # (No specific setup required)

        # When
        with patch.object(QDialog, 'exec_') as mock_exec:
            self.viewer.on_about()

        # Then
        mock_exec.assert_called()

    def test_on_slider_move(self):
        """
        Checks whether function 'on_slider_move' changes pixmaps as sliders move.
        """
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

        # When & Then
        try:
            self.viewer.on_slider_move()
        except Exception as e:
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


    def test_render_3d_visualization(self):
        """
        New test: checks the 'render_3d_visualization' method with a mock segmentation file.
        """
        # Given
        with patch.object(self.viewer, 'log_message'), \
             patch.object(self.viewer, 'vtk_widget', MagicMock()), \
             patch('GUI.nib.load') as mock_nib_load, \
             patch('demo.convert_to_stl'):
            mock_nifti = MagicMock()
            mock_nifti.get_fdata.return_value = np.zeros((5, 5, 5), dtype=np.uint8)
            mock_nib_load.return_value = mock_nifti
            self.viewer.render_3d_visualization(seg_file='mock_seg.nii.gz')

            # Then
            # As in render_3d_visualization, if it has data, it tries to do the 3D steps
            # We'll just confirm no exceptions happen
            # For thoroughness:
            #  - We can't do a strong assertion if all is stubbed, but at least no crash:
            self.assertTrue(True, "render_3d_visualization executed without crashing.")

    def test_on_manage_view(self):
        """
        New test: checks whether 'on_manage_view' opens the organ selection dialog, and whether the
        user selection triggers update_3d_view call.
        """
        # Given
        # Mock data for segmentation_result
        seg_data = np.zeros((10, 10, 10), dtype=np.uint8)
        seg_data[2:4, 2:4, 2:4] = 1  # label=1
        seg_data[5:7, 5:7, 5:7] = 2  # label=2
        self.viewer.segmentation_result = seg_data

        # When
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('spleen', True)), \
             patch.object(self.viewer, 'update_3d_view') as mock_update_3d_view:
            self.viewer.on_manage_view()

        # Then
        mock_update_3d_view.assert_called()

    def test_update_3d_view(self):
        """
        New test: checks whether 'update_3d_view' adjusts alpha on volumes as expected.
        """
        # Given
        # create two mock volumes in self.loaded_volumes
        mock_volume1 = MagicMock()
        mock_volume2 = MagicMock()
        self.viewer.loaded_volumes = {
            'Organ1': mock_volume1,
            'Organ2': mock_volume2,
        }
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        # Show only 'Organ1'
        self.viewer.update_3d_view(['Organ1'])

        # Then
        mock_volume1.alpha.assert_called_with(1)  # visible
        mock_volume2.alpha.assert_called_with(0)  # hidden
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_on_calculate_volume_no_segmentation(self):
        """
        New test: checks whether function 'on_calculate_volume' warns if there's no segmentation.
        """
        # Given
        self.viewer.segmentation_result = None
        self.viewer.affine = None

        # When
        with patch.object(QMessageBox, 'warning') as mock_warning:
            self.viewer.on_calculate_volume()

        # Then
        mock_warning.assert_called()

    def test_on_calculate_volume_with_data(self):
        """
        New test: checks whether function 'on_calculate_volume' calls volume calculation logic
        after user selects an organ from the QInputDialog.
        """
        # Given
        # create a segmentation with label=1
        seg_data = np.zeros((10, 10, 10), dtype=np.uint8)
        seg_data[2:4, 2:4, 2:4] = 1  # a small block
        self.viewer.segmentation_result = seg_data
        self.viewer.affine = np.eye(4)
        self.viewer.current_segmented_labels = np.array([1], dtype=np.uint8)

        # When
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('spleen', True)), \
             patch.object(self.viewer, 'calculate_volume_for_organ') as mock_calc:
            self.viewer.on_calculate_volume()

        # Then
        mock_calc.assert_called_with('spleen')

    def test_calculate_volume_for_organ_no_voxels(self):
        """
        New test: checks whether 'calculate_volume_for_organ' handles the case when the organ
        has no voxels in the current segmentation.
        """
        # Given
        # label 1 does not appear in segmentation
        self.viewer.segmentation_result = np.zeros((10, 10, 10), dtype=np.uint8)
        self.viewer.affine = np.eye(4)

        # Suppose 'spleen' -> label=1 in your dictionary

        # When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.calculate_volume_for_organ('spleen')

        # Then
        mock_info.assert_called()
        self.assertIn("No voxels found for spleen", mock_info.call_args[0][2])

    def test_calculate_volume_for_organ_with_voxels(self):
        """
        New test: checks whether 'calculate_volume_for_organ' calculates volume if voxels exist.
        """
        # Given
        # label 1 -> has some voxels
        seg_data = np.zeros((10, 10, 10), dtype=np.uint8)
        seg_data[2:5, 2:5, 2:5] = 1
        self.viewer.segmentation_result = seg_data
        self.viewer.affine = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)
        # This means each voxel is 1x1x1 mm => 1 mm^3 per voxel

        # When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.calculate_volume_for_organ('spleen')

        # Then
        mock_info.assert_called()
        # check if message about volume was displayed
        self.assertIn("The volume for spleen is approximately", mock_info.call_args[0][2])



    def test_on_contribute(self):
        """
        New test: checks whether function 'on_contribute' opens the GitHub URL.
        """
        # Given
        # (No specific setup required)

        # When
        with patch('GUI.QDesktopServices.openUrl') as mock_open_url:
            self.viewer.on_contribute()

        # Then
        mock_open_url.assert_called_once()

    def test_create_3d_visualization_window(self):
        """
        New test: checks whether function 'create_3d_visualization_window' can be called
        without throwing exceptions.
        """
        # Given
        seg_data = np.zeros((5, 5, 5), dtype=np.uint8)
        seg_data[2, 2, 2] = 1

        # When
        with patch.object(self.viewer, 'log_message'), \
             patch('GUI.Plotter') as mock_plotter_class, \
             patch('demo.convert_to_stl'):
            try:
                self.viewer.create_3d_visualization_window(seg_data)
                # if no crash => success
            except Exception as e:
                self.fail(f"create_3d_visualization_window raised an exception: {e}")

        # Then
        # (No assertions needed as we only check for absence of exceptions)


class TestPerformance(unittest.TestCase):
    """ Tests for performance (timing). """

    def test_full_pipeline(self):
        """
        Checks the operation of the full processing path: preprocessing, segmentation, saving results.
        """
        # Given
        ct_scan = np.random.rand(128, 128, 128).astype(np.float32)
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

        # Cleanup
        os.remove(nifti_path)
        os.remove("temp_output.nii.gz")

    def test_segmentation_speed(self):
        """
        Checks the speed of preprocessing and segmentation, ensuring they are below certain thresholds.
        """
        # Given
        ct_scan = np.random.rand(128, 128, 128).astype(np.float32)

        # When
        start_preprocess = time.time()
        processed_scan = preprocess_img(ct_scan, target_size=(128, 128, 128))
        end_preprocess = time.time()
        preprocess_time = end_preprocess - start_preprocess
        print(f"Preprocessing time: {preprocess_time:.4f} seconds")

        unet_model = get_unet_model(num_classes=2, in_channels=1)
        unet_model.eval()
        start_segmentation = time.time()
        segmentation_result = segment_img(unet_model, processed_scan)
        end_segmentation = time.time()
        segmentation_time = end_segmentation - start_segmentation
        print(f"Segmentation time: {segmentation_time:.4f} seconds")

        # Then
        self.assertLess(preprocess_time, 5.0, "Preprocessing time exceeded 5 seconds")
        self.assertLess(segmentation_time, 10.0, "Segmentation time exceeded 10 seconds")


def test_visualization_speed():
    """
    Standalone test function: checks the speed of simple 'display_single_slice'.
    """
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
    assert visualization_time < 4.0, "Visualization time exceeded 4 seconds"


if __name__ == '__main__':
    unittest.main()
