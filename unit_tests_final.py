# === unit_tests.py ===

import unittest
import os
import sys
import time
import numpy as np
import nibabel as nib
import torch

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMessageBox, QDialog, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtCore import Qt
from unittest.mock import patch, MagicMock

# Project imports
import data_import
import visualization
import segmentation
import demo
from GUI import CTViewer, ZoomCT

_app = None

def setUpModule():
    """
    This setUpModule runs once at the start of all tests in this module.
    """
    global _app
    if not QApplication.instance():
        _app = QApplication(sys.argv)

def tearDownModule():
    """
    Runs once at the end of all tests in this module.
    """
    global _app
    if _app:

        _app.quit()
        _app = None


###############################################################################
#                          T E S T   D A T A   I M P O R T
###############################################################################

class TestDataImport(unittest.TestCase):
    """Tests for data_import.py"""

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
        os.remove(temp_file)

    def test_validate_nifti_invalid(self):
        """
        Checks whether function 'validate_nifti' correctly identifies an invalid file (like .txt).
        """
        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('Not a NIfTI file.')

        # When
        result = data_import.validate_nifti(temp_file)

        # Then
        self.assertFalse(result)
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
        os.remove(temp_file)


###############################################################################
#                        T E S T   V I S U A L I Z A T I O N
###############################################################################

class TestVisualization(unittest.TestCase):
    """Tests for visualization.py"""



    def test_convert_img_slices(self):
        """
        Checks whether 'convert_img_slices' correctly splits a NIfTI file into scans.
        """
        # Given
        data = np.random.rand(50, 50, 50).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        temp_file = 'temp_nifti.nii.gz'
        nib.save(nifti_img, temp_file)

        # When
        sag_list, cor_list, ax_list, loaded_affine = visualization.convert_img_slices(temp_file)

        # Then
        self.assertEqual(len(sag_list), 128)  # default param => 128 slices in sagittal
        self.assertEqual(len(cor_list), 300)  # 300 in coronal
        self.assertEqual(len(ax_list), 400)   # 400 in axial
        os.remove(temp_file)

    def test_convert_img_slices_invalid_file(self):
        """
        Checks the response of 'convert_img_slices' to invalid NIfTI files.
        """
        # Given
        temp_file = 'temp_invalid_nifti.txt'
        with open(temp_file, 'w') as f:
            f.write('Invalid data')

        # When & Then
        with self.assertRaises(ValueError):
            visualization.convert_img_slices(temp_file)
        os.remove(temp_file)

    def test_display_single_slice(self):
        """
        Checks whether 'display_single_slice' displays one slice in a QLabel.
        """
        # Given
        label = QLabel()
        label.setFixedSize(400, 300)
        data = np.random.rand(300, 400) * 255
        data = data.astype(np.uint8)
        qimg = QImage(
            data.data.tobytes(),
            data.shape[1],
            data.shape[0],
            QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(qimg)

        # When
        visualization.display_single_slice(label, pixmap)

        # Then
        self.assertIsNotNone(label.pixmap())

    def test_overlay_slices(self):
        """
        Checks whether 'overlay_slices' overlays mask on base image.
        """
        # Given
        w, h = 48, 48
        base_array = np.full((h, w), 100, dtype=np.uint8)
        mask_array = np.zeros((h, w), dtype=np.uint8)
        mask_array[10:20, 10:20] = 1  # label=1

        base_qimg = QImage(
            base_array.tobytes(),
            w,
            h,
            w,  # <-- bytesPerLine = width * 1 byte/pixel
            QImage.Format_Grayscale8
        )

        base_pixmap = QPixmap.fromImage(base_qimg)

        mask_qimg = QImage(
            mask_array.tobytes(),
            w,
            h,
            w,
            QImage.Format_Grayscale8
        )

        mask_pixmap = QPixmap.fromImage(mask_qimg)
        class_to_color = {1: '#FF0000'}

        # When
        try:
            result = visualization.overlay_slices(base_pixmap, mask_pixmap, alpha=0.5, class_to_color=class_to_color)
        except ValueError as e:
            self.fail(f"overlay_slices raised an exception: {e}")

        #  Then
        self.assertIsInstance(result, QPixmap)
        self.assertFalse(result.isNull())


###############################################################################
#                       T E S T   S E G M E N T A T I O N
###############################################################################

class TestSegmentation(unittest.TestCase):
    """Tests for segmentation.py"""

    def test_preprocess_img(self):
        """
        Checks whether 'preprocess_img' transforms a numpy array into the correct tensor shape.
        """
        # Given
        data = np.random.rand(128, 128, 128).astype(np.float32)

        # When
        tensor_out = segmentation.preprocess_img(data)

        # Then
        self.assertEqual(tensor_out.shape, (1, 128, 128, 128))
        self.assertTrue(isinstance(tensor_out, torch.Tensor))

    def test_save_segmentation(self):
        """
        Checks whether 'save_segmentation' properly saves data to a NIfTI file.
        """
        # Given
        seg_data = np.random.randint(0, 2, (64, 64, 64)).astype(np.int16)
        affine = np.eye(4)
        out_path = 'temp_segmentation.nii.gz'

        # When
        segmentation.save_segmentation(seg_data, affine, out_path)

        # Then
        self.assertTrue(os.path.exists(out_path))
        loaded = nib.load(out_path).get_fdata()
        # Since get_fdata() returns float data, cast seg_data to float for comparison
        np.testing.assert_array_equal(seg_data.astype(np.float32), loaded)
        os.remove(out_path)


###############################################################################
#                           T E S T   D E M O
###############################################################################

class TestDemo(unittest.TestCase):
    """Tests for demo.py"""

    def test_convert_to_stl(self):
        """
        Checks whether 'convert_to_stl' writes an STL file for non-empty data.
        """
        # Given
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[2:6, 2:6, 2:6] = 1
        out_file = 'temp_mesh.stl'

        # When
        demo.convert_to_stl(data, out_file)

        # Then
        self.assertTrue(os.path.exists(out_file))
        os.remove(out_file)

    def test_convert_to_stl_empty_data(self):
        """
        Checks whether 'convert_to_stl' does not create a file for empty data.
        """
        # Given
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        out_file = 'temp_empty_mesh.stl'

        # When
        demo.convert_to_stl(data, out_file)

        # Then
        self.assertFalse(os.path.exists(out_file))


###############################################################################
#                          T E S T   Z O O M C T
###############################################################################

class TestZoomCT(unittest.TestCase):
    """Tests for the ZoomCT (double-click zoom)."""

    def setUp(self):
        """
        #Given a ZoomCT label with a small pixmap
        """

        self.zoom_label = ZoomCT()
        test_pixmap = QPixmap(100, 100)
        self.zoom_label.setPixmap(test_pixmap)

    def test_mouse_double_click_event(self):
        """
        #Given: the label has a pixmap
        #When: user double clicks
        #Then: a dialog should open with the zoomed image
        """
        # When
        with patch('PyQt5.QtWidgets.QDialog.exec_', return_value=None) as mock_dialog:
            event = QMouseEvent(
                QMouseEvent.MouseButtonDblClick,
                self.zoom_label.rect().center(),
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier
            )
            self.zoom_label.mouseDoubleClickEvent(event)

        # Then
        mock_dialog.assert_called_once()


###############################################################################
#                             T E S T   G U I
###############################################################################

class TestGUI(unittest.TestCase):
    """Tests for the CTViewer class in GUI.py."""

    def setUp(self):


        self.viewer = CTViewer()

    def test_log_message(self):
        """
        Checks whether log_message writes to the error_log.
        """
        # Given
        self.viewer.error_log = MagicMock()

        # When
        self.viewer.log_message("Test message")

        # Then
        self.viewer.error_log.appendPlainText.assert_called_with("Test message")

    def test_init_ui(self):
        """
        Checks init_ui for any exceptions.
        """
        # Given & When
        try:
            self.viewer.init_ui()
        except Exception as e:
            # Then
            self.fail(f"init_ui raised an exception: {e}")

    def test_setup_menu_bar(self):
        """
        Checks that the menu bar has the expected actions.
        """
        # Given
        self.viewer.setup_menu_bar()

        # When
        menu_bar = self.viewer.menuBar()
        file_menu = menu_bar.actions()[0].menu()
        actions = [act.text() for act in file_menu.actions()]

        # Then
        self.assertIn('Upload data', actions)
        self.assertIn('Save segmentation', actions)
        self.assertIn('Close segmentation', actions)

    def test_upload_data_cancel(self):
        """
        Checks whether no method is called if user cancels uploading data.
        """
        # Given
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('', False)):
            # When
            with patch.object(self.viewer, 'upload_ct_scan') as mock_upload_ct, \
                 patch.object(self.viewer, 'upload_segmented_ct_scan') as mock_upload_seg:
                self.viewer.upload_data()

        # Then
        mock_upload_ct.assert_not_called()
        mock_upload_seg.assert_not_called()

    def test_upload_ct_scan_valid(self):
        """
        Checks upload_ct_scan with a valid file sets ct_scans and enables segmentation.
        """
        # Given
        file_selection = ['valid.nii']
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(file_selection[0], 'NIfTI Files (*.nii *.nii.gz)')), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((10,10,10)), np.eye(4))), \
             patch('visualization.convert_img_slices', return_value=([], [], [], np.eye(4))), \
             patch.object(self.viewer, 'update_img_placeholders'), \
             patch.object(self.viewer, 'render_3D_visualization'):
            # When
            self.viewer.upload_ct_scan()

        # Then
        self.assertIsNotNone(self.viewer.ct_scans)
        self.assertTrue(self.viewer.segment_action.isEnabled())

    def test_upload_ct_scan_invalid(self):
        """
        Checks that a critical message is shown if file is invalid.
        """
        # Given
        file_selection = 'invalid.nii'

        # When
        with patch('PyQt5.QtWidgets.QFileDialog.exec_', return_value=True), \
             patch('PyQt5.QtWidgets.QFileDialog.selectedFiles', return_value=file_selection), \
             patch('data_import.validate_nifti', return_value=False), \
             patch.object(QMessageBox, 'critical') as mock_critical:
            self.viewer.upload_ct_scan()

        # Then
        mock_critical.assert_called()


    def test_upload_segmented_ct_scan_valid(self):
        """
        Checks that seg_result is set and segmentation is disabled if already segmented.
        """
        # Given
        file_selection = ['segmented.nii']
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(file_selection[0], 'NIfTI Files (*.nii *.nii.gz)')), \
             patch('data_import.validate_nifti', return_value=True), \
             patch('data_import.load_nifti', return_value=(np.zeros((5,5,5)), np.eye(4))), \
             patch('visualization.convert_seg_slices', return_value=([], [], [], np.eye(4))), \
             patch.object(self.viewer, 'render_3D_visualization_seg'):
            # When
            self.viewer.upload_segmented_ct_scan()

        # Then
        self.assertIsNotNone(self.viewer.seg_result)
        self.assertFalse(self.viewer.segment_action.isEnabled())

    def test_segment_image_no_ct_scans(self):
        """
        Checks if a warning is displayed if no CT scans are loaded.
        """
        # Given
        self.viewer.ct_scans = None

        # When
        with patch.object(QMessageBox, 'warning') as mock_warn:
            self.viewer.segment_image()

        # Then
        mock_warn.assert_called()

    def test_segment_image_ok(self):
        """
        Checks normal flow of segment_image with a mock segmentation call.
        """
        # Given
        self.viewer.ct_scans = np.zeros((10,10,10))
        self.viewer.file_path = 'dummy.nii'

        # When
        with patch('segmentation.segment_img_with_TS') as mock_seg, \
             patch('visualization.convert_seg_slices', return_value=([], [], [])), \
             patch.object(self.viewer, 'render_3D_visualization_seg'):
            mock_seg.return_value = MagicMock()
            mock_seg.return_value.get_fdata.return_value = np.zeros((10,10,10))
            self.viewer.segment_image()

        # Then
        self.assertIsNotNone(self.viewer.seg_result)

    def test_update_img_placeholders(self):
        """
        Checks that the placeholders are updated with the first slice from each direction.
        """
        # Given
        self.viewer.scan_list_sagittal = [QPixmap(100,100)]
        self.viewer.scan_list_coronal = [QPixmap(100,100)]
        self.viewer.scan_list_axial = [QPixmap(100,100)]

        # When
        self.viewer.update_img_placeholders()

        # Then
        self.assertIsNotNone(self.viewer.scan_top_left.pixmap())
        self.assertIsNotNone(self.viewer.scan_top_right.pixmap())
        self.assertIsNotNone(self.viewer.scan_bottom_left.pixmap())

    def test_save_segmentation_with_data(self):
        """
        Checks that segmentation is saved if seg_result is present.
        """
        # Given
        self.viewer.seg_result = np.zeros((10, 10, 10))

        # When
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=('seg_result.nii.gz', '')), \
             patch('segmentation.save_segmentation') as mock_save, \
             patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.save_segmentation()

        # Then
        mock_save.assert_called_with(
            self.viewer.seg_result,
            self.viewer.affine,
            'seg_result.nii.gz'
        )
        mock_info.assert_called()

    def test_save_segmentation_no_data(self):
        """
        Checks that a message is shown if no seg_result to save.
        """
        # Given
        self.viewer.seg_result = None

        # When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.save_segmentation()

        # Then
        mock_info.assert_called()
        args = mock_info.call_args[0]
        self.assertIn("There is no segmentation data to save.", args[2])

    def test_close_segmentation(self):
        """
        Checks whether 'close_segmentation' resets state to initial.
        """
        # Given
        self.viewer.ct_scans = np.zeros((10,10,10))
        self.viewer.affine = np.eye(4)
        self.viewer.seg_result = np.zeros((10,10,10))
        self.viewer.seg_sagittal = [QPixmap(100,100)]
        self.viewer.seg_coronal = [QPixmap(100,100)]
        self.viewer.seg_axial = [QPixmap(100,100)]
        self.viewer.scan_list_sagittal = [QPixmap(100,100)]
        self.viewer.scan_list_coronal = [QPixmap(100,100)]
        self.viewer.scan_list_axial = [QPixmap(100,100)]
        self.viewer.scan_top_left = QLabel()
        self.viewer.scan_top_right = QLabel()
        self.viewer.scan_bottom_left = QLabel()
        self.viewer.slider_sagittal.setMaximum(10)
        self.viewer.slider_coronal.setMaximum(10)
        self.viewer.slider_axial.setMaximum(10)
        self.viewer.slider_sagittal.setValue(5)
        self.viewer.slider_coronal.setValue(5)
        self.viewer.slider_axial.setValue(5)
        self.viewer.loaded_vol = {'Organ1': MagicMock(), 'Organ2': MagicMock()}
        self.viewer.plotter = MagicMock()

        # When
        self.viewer.close_segmentation()

        # Then
        self.assertIsNone(self.viewer.ct_scans)
        self.assertIsNone(self.viewer.affine)
        self.assertIsNone(self.viewer.seg_result)
        self.assertEqual(self.viewer.seg_sagittal, [])
        self.assertEqual(self.viewer.seg_coronal, [])
        self.assertEqual(self.viewer.seg_axial, [])
        self.assertEqual(self.viewer.scan_list_sagittal, [])
        self.assertEqual(self.viewer.scan_list_coronal, [])
        self.assertEqual(self.viewer.scan_list_axial, [])
        self.assertFalse(self.viewer.segment_action.isEnabled())

    def test_manage_view_no_seg_result(self):
        """
        Checks if 'manage_view' warns the user if seg_result is None or empty.
        """
        # Given
        self.viewer.seg_result = None

        # When
        with patch.object(QMessageBox, 'warning') as mock_warn:
            self.viewer.manage_view()

        # Then
        mock_warn.assert_called()

    def test_manage_view_ok(self):
        """
        Checks if 'manage_view' calls update_3D_view after picking an organ in the dialog.
        """
        # Given
        seg_data = np.zeros((10,10,10), dtype=np.uint8)
        seg_data[2:4,2:4,2:4] = 1
        seg_data[5:7,5:7,5:7] = 2
        self.viewer.seg_result = seg_data

        # When
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('spleen', True)), \
             patch.object(self.viewer, 'update_3D_view') as mock_update:
            self.viewer.manage_view()

        # Then
        mock_update.assert_called_with(['spleen'])

    def test_update_3D_view(self):
        """
        Checks whether 'update_3D_view' changes alpha for selected organs.
        """
        # Given
        mock_vol1 = MagicMock()
        mock_vol2 = MagicMock()
        self.viewer.loaded_vol = {
            'Organ1': mock_vol1,
            'Organ2': mock_vol2
        }
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        self.viewer.update_3D_view(['Organ1'])

        # Then
        mock_vol1.alpha.assert_called_with(1)
        mock_vol2.alpha.assert_called_with(0)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_calculate_volume_no_seg(self):
        """
        Checks if 'calculate_volume' warns user if no seg_result or affine.
        """
        # Given
        self.viewer.seg_result = None
        self.viewer.affine = None

        # When
        with patch.object(QMessageBox, 'warning') as mock_warn:
            self.viewer.calculate_volume()

        # Then
        mock_warn.assert_called()

    def test_calculate_volume_with_data(self):
        """
        Checks if 'calculate_volume' calls 'calculate_volume_for_organ' for chosen organ.
        """
        # Given
        seg_data = np.zeros((10,10,10), dtype=np.uint8)
        seg_data[1:2,1:2,1:2] = 1
        self.viewer.seg_result = seg_data
        self.viewer.affine = np.eye(4)
        self.viewer.curr_seg_lbls = np.array([1], dtype=np.uint8)

        # When
        with patch('PyQt5.QtWidgets.QInputDialog.getItem', return_value=('spleen', True)), \
             patch.object(self.viewer, 'calculate_volume_for_organ') as mock_calc:
            self.viewer.calculate_volume()

        # Then
        mock_calc.assert_called_with('spleen')

    def test_calculate_volume_for_organ_no_voxels(self):
        """
        Checks if 'calculate_volume_for_organ' shows message if no voxels found.
        """
        # Given
        self.viewer.seg_result = np.zeros((10,10,10), dtype=np.uint8)
        self.viewer.affine = np.eye(4)

        # When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.calculate_volume_for_organ('spleen')

        # Then
        mock_info.assert_called()
        self.assertIn("No voxels found for spleen", mock_info.call_args[0][2])

    def test_calculate_volume_for_organ_with_voxels(self):
        """
        Checks if 'calculate_volume_for_organ' calculates volume for organ with voxels.
        """
        # Given
        seg_data = np.zeros((10,10,10), dtype=np.uint8)
        seg_data[2:5,2:5,2:5] = 1
        self.viewer.seg_result = seg_data
        self.viewer.affine = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)

        # When
        with patch.object(QMessageBox, 'information') as mock_info:
            self.viewer.calculate_volume_for_organ('spleen')

        # Then
        mock_info.assert_called()
        self.assertIn("The volume for spleen is approximately", mock_info.call_args[0][2])

    def test_zoom_in(self):
        """
        Checks if 'zoom_in' calls plotter.zoom(1.1).
        """
        # Given
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        self.viewer.zoom_in()

        # Then
        self.viewer.plotter.zoom.assert_called_with(1.1)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_zoom_out(self):
        """
        Checks if 'zoom_out' calls plotter.zoom(0.9).
        """
        # Given
        self.viewer.plotter = MagicMock()
        self.viewer.vtk_widget = MagicMock()

        # When
        self.viewer.zoom_out()

        # Then
        self.viewer.plotter.zoom.assert_called_with(0.9)
        self.viewer.plotter.render.assert_called()
        self.viewer.vtk_widget.update.assert_called()

    def test_report_problem(self):
        """
        Checks if 'report_problem' opens the default mail client.
        """
        # Given & When
        with patch('PyQt5.QtGui.QDesktopServices.openUrl') as mock_open:
            self.viewer.report_problem()

        # Then
        mock_open.assert_called_once()

    def test_help(self):
        """
        Checks if 'help' shows a QMessageBox.
        """
        # Given & When
        with patch.object(QMessageBox, 'exec_') as mock_exec:
            self.viewer.help()

        # Then
        mock_exec.assert_called()

    def test_about(self):
        """
        Checks if 'about' opens a QDialog.
        """
        # Given & When
        with patch.object(QDialog, 'exec_') as mock_dlg:
            self.viewer.about()

        # Then
        mock_dlg.assert_called()

    def test_slider_move(self):
        """
        Checks if 'slider_move' updates slices with overlay if available.
        """
        # Given
        self.viewer.scan_list_sagittal = [QPixmap(50, 50) for _ in range(5)]
        self.viewer.scan_list_coronal = [QPixmap(50, 50) for _ in range(5)]
        self.viewer.scan_list_axial = [QPixmap(50, 50) for _ in range(5)]
        self.viewer.seg_sagittal = [QPixmap(50, 50) for _ in range(5)]
        self.viewer.seg_coronal = [QPixmap(50, 50) for _ in range(5)]
        self.viewer.seg_axial = [QPixmap(50, 50) for _ in range(5)]

        self.viewer.scan_top_left = QLabel()
        self.viewer.scan_top_right = QLabel()
        self.viewer.scan_bottom_left = QLabel()

        self.viewer.slider_sagittal.setMaximum(4)
        self.viewer.slider_coronal.setMaximum(4)
        self.viewer.slider_axial.setMaximum(4)
        self.viewer.slider_sagittal.setValue(2)
        self.viewer.slider_coronal.setValue(2)
        self.viewer.slider_axial.setValue(2)

        # When
        try:
            self.viewer.slider_move()
        except Exception as e:
            self.fail(f"slider_move raised an exception: {e}")

        # Then
        # No direct assertion beyond "no crash". We assume overlay is displayed.


###############################################################################
#                   T E S T   P E R F O R M A N C E
###############################################################################

class TestPerformance(unittest.TestCase):
    """Tests for performance/timing."""

    def test_full_pipeline(self):
        """
        Checks that a .nii.gz can be saved successfully.
        """
        # Given
        ct_scan = np.random.rand(128, 128, 128).astype(np.float32)
        affine = np.eye(4)
        in_path = "temp_ct.nii.gz"
        nib.save(nib.Nifti1Image(ct_scan, affine), in_path)

        # When
        out_path = "temp_output.nii.gz"
        dummy_seg = np.zeros((128,128,128), dtype=np.int16)
        segmentation.save_segmentation(dummy_seg, affine, out_path)

        # Then
        self.assertTrue(os.path.exists(out_path))
        os.remove(in_path)
        os.remove(out_path)

    def test_segment_image_speed_totalsegmentator(self):
        """
        Checks that segment_image (with totalsegmentator) completes under 40s (mocked).
        """
        # Given
        viewer = CTViewer()
        viewer.file_path = 'dummy.nii'
        viewer.ct_scans = np.zeros((128,128,128), dtype=np.float32)

        # When
        start = time.time()
        with patch('segmentation.segment_img_with_TS') as mock_seg:
            time.sleep(0.5)  # simulate some load
            mock_seg.return_value = MagicMock()
            mock_seg.return_value.get_fdata.return_value = np.zeros((128,128,128))
            viewer.segment_image()
        elapsed = time.time() - start

        # Then
        self.assertLess(elapsed, 40.0, f"Segmentation took {elapsed:.2f}s, exceeds 40s limit!")


################################################################################
#               A  S T A N D A L O N E   F U N C T I O N   T E S T
################################################################################

def test_visualization_speed():
    """
    Standalone test function checking display_single_slice speed.
    """
    # Given
    label = QLabel()
    label.setFixedSize(400, 300)
    pixmap = QPixmap(128, 128)

    start_t = time.time()
    # When
    for _ in range(10):
        visualization.display_single_slice(label, pixmap)
    end_t = time.time()

    # Then
    duration = end_t - start_t
    assert duration < 4.0, f"display_single_slice took {duration:.2f}s, exceeds 4s"


if __name__ == '__main__':
    unittest.main()
