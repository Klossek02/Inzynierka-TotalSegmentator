
# === GUI.py ===

import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor
import os
import requests
import numpy as np
import nibabel as nib
import torch
from monai.transforms import Compose, Resize
import demo

# saving predicted 3D volume as a NIfTI file
def save_nifti(volume, path, index=0):
    volume = np.array(volume.detach().cpu()[0], dtype=np.float32)
    volume = nib.Nifti1Image(volume, np.eye(4))
    nib.save(volume, os.path.join(path, f'patient_predicted_{index}.nii.gz'))
    print(f'patient_predicted_{index} is saved', end='\r')

# loading and processing a NIfTI file, transforming it to a 2D image slices 
def convert_nifti(file_name, logger=None):
    try:
        if logger:
            logger(f"Loading NIfTI file: {file_name}")

        scan = nib.load(file_name)
        scan_array = scan.get_fdata() # extracting a raw data from the scan 

        # examining and getting the scan's shape and header for dimensions
        scan_array_shape = scan_array.shape
        scan_header = scan.header
        pix_dim = scan_header['pixdim'][1:4] # extracting pixel dimensions for scaling 

        # calculating new image dimensions from the aspect ratio
        new_scan_dims = np.multiply(scan_array_shape, pix_dim)
        new_scan_dims = (round(new_scan_dims[0]), round(new_scan_dims[1]), round(new_scan_dims[2]))

        # initlizing lists to store 2D slices for each orientation
        output_array0, output_array1,output_array2 = [], [], []

        if logger:
            logger("Converting slices...")

        # creating 2D slices along the x-axis
        for i in range(scan_array_shape[0]):
            output_array = cv2.resize(scan_array[i, :, :], (new_scan_dims[2], new_scan_dims[1]))
            cv2.imwrite('ex_slice_1.jpeg', output_array)
            pixmap = QPixmap('ex_slice_1.jpeg')
            output_array0.append(pixmap)
            if logger and i % 10 == 0:
                logger(f"Converted slice {i + 1}/{scan_array_shape[0]} in dimension 0")

        # creating 2D slices along the y-axis
        for i in range(scan_array_shape[1]):
            output_array = cv2.resize(scan_array[:, i, :], (new_scan_dims[2], new_scan_dims[0]))
            cv2.imwrite('ex_slice_1.jpeg', output_array)
            pixmap = QPixmap('ex_slice_1.jpeg')
            output_array1.append(pixmap)
            if logger and i % 10 == 0:
                logger(f"Converted slice {i + 1}/{scan_array_shape[1]} in dimension 1")

        # creating 2D slices along the z-axis
        for i in range(scan_array_shape[2]):
            output_array = cv2.resize(scan_array[:, :, i], (new_scan_dims[1], new_scan_dims[0]))
            cv2.imwrite('ex_slice_1.jpeg', output_array)
            pixmap = QPixmap('ex_slice_1.jpeg')
            output_array2.append(pixmap)
            if logger and i % 10 == 0:
                logger(f"Converted slice {i + 1}/{scan_array_shape[2]} in dimension 2")

        if logger:
            logger("Slices converted successfully.")
        return output_array0, output_array1, output_array2
    except Exception as e:
        error_message = f"Error in convert_nifti: {str(e)}"
        if logger:
            logger(error_message)
        else:
            print(error_message)
        raise e

# main application window for viewing and interacting with medical images
class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    # log messages send to the error log panel in the UI
    def log_message(self, message):
        self.error_log.appendPlainText(message)

    # setting up initial UI components
    def init_ui(self):
        self.setWindowTitle('SegMed 1.2')
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        QToolTip.setFont(QFont('SansSerif', 10))

        # screen dimensions
        screen = QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        self.setMaximumSize(screen_width, screen_height)

        # applying style to wigets
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: #fff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #eee;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #66b3ff;
                border: 1px solid #3399ff;
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QPlainTextEdit {
                background-color: #2b2b2b;
                color: #a9b7c6;
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                border: none;
            }
            QPushButton {
                background-color: #66b3ff;
                color: #fff;
                border-radius: 10px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #5599e6;
            }
            QMenuBar {
                background-color: #fff;
            }
            QMenuBar::item {
                background-color: #fff;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #f0f0f0;
            }
            QMenu {
                background-color: #fff;
            }
            QMenu::item:selected {
                background-color: #66b3ff;
                color: #fff;
            }
        """)

        self.setup_menu_bar() # setup menu bar with options 

        # main layout - for controls and images 
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # grid layout - for image placeholders and sliders
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignCenter)

        self.image_collection0 = []
        self.image_collection1 = []
        self.image_collection2 = []

        # image placeholders
        self.image_placeholder = QLabel()
        self.image_placeholder.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder.setAlignment(Qt.AlignCenter)
        self.image_placeholder.setStyleSheet("border: 1px solid #ccc; border-radius: 10px; background-color: #fff;")
        self.image_placeholder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_placeholder.setFixedSize(400, 300)

        self.image_placeholder1 = QLabel()
        self.image_placeholder1.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder1.setAlignment(Qt.AlignCenter)
        self.image_placeholder1.setStyleSheet("border: 1px solid #ccc; border-radius: 10px; background-color: #fff;")
        self.image_placeholder1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_placeholder1.setFixedSize(400, 300)

        self.image_placeholder2 = QLabel()
        self.image_placeholder2.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder2.setAlignment(Qt.AlignCenter)
        self.image_placeholder2.setStyleSheet("border: 1px solid #ccc; border-radius: 10px; background-color: #fff;")
        self.image_placeholder2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_placeholder2.setFixedSize(400, 300)

        # vtk widget - used for 3D rendering 
        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.vtk_widget.setFixedSize(400, 300)

        # sliders - to scroll through image slices 
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_move)
        self.slider.setStyleSheet("QSlider { margin-top: 10px; }")
        self.slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slider.setFixedWidth(400)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.valueChanged.connect(self.on_slider_move)
        self.slider1.setStyleSheet("QSlider { margin-top: 10px; }")
        self.slider1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slider1.setFixedWidth(400)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.valueChanged.connect(self.on_slider_move)
        self.slider2.setStyleSheet("QSlider { margin-top: 10px; }")
        self.slider2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slider2.setFixedWidth(400)

        inner_layout = QVBoxLayout()
        inner_layout.addWidget(self.image_placeholder)
        inner_layout.addWidget(self.slider)
        inner_layout.setAlignment(Qt.AlignCenter)

        inner1_layout = QVBoxLayout()
        inner1_layout.addWidget(self.image_placeholder1)
        inner1_layout.addWidget(self.slider1)
        inner1_layout.setAlignment(Qt.AlignCenter)

        inner2_layout = QVBoxLayout()
        inner2_layout.addWidget(self.image_placeholder2)
        inner2_layout.addWidget(self.slider2)
        inner2_layout.setAlignment(Qt.AlignCenter)

        inner3_layout = QVBoxLayout()
        inner3_layout.addWidget(self.vtk_widget)
        inner3_layout.setAlignment(Qt.AlignCenter)

        grid_layout.addLayout(inner_layout, 0, 0)
        grid_layout.addLayout(inner1_layout, 0, 1)
        grid_layout.addLayout(inner2_layout, 1, 0)
        grid_layout.addLayout(inner3_layout, 1, 1)

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        grid_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        grid_widget.adjustSize()

        main_layout.addWidget(grid_widget, alignment=Qt.AlignCenter)

        # error log - to display any log or error messages
        self.error_log = QPlainTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setFixedHeight(150)
        self.error_log.appendPlainText("Error Log:\n")
        self.error_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # setting central wiget
        main_layout.addWidget(self.error_log)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        try:
            self.log_message("Loading example images...")
            self.image_collection0.append(QPixmap('ex_slice_1.jpeg'))
            self.image_placeholder.setPixmap(self.image_collection0[0])
            self.image_collection1.append(QPixmap('ex_slice_1.jpeg'))
            self.image_placeholder1.setPixmap(self.image_collection1[0])
            self.image_collection2.append(QPixmap('ex_slice_1.jpeg'))
            self.image_placeholder2.setPixmap(self.image_collection2[0])
            self.log_message("Example images loaded successfully.")
        except Exception as e:
            error_message = f"Error loading example images: {str(e)}"
            self.log_message(error_message)

        self.render_3d_visualization()

        self.adjustSize()
        window_size = self.size()
        if window_size.width() > screen_width or window_size.height() > screen_height:
            self.resize(screen_width, screen_height)
        self.move((screen_width - self.width()) // 2, (screen_height - self.height()) // 2)
        self.log_message("UI initialized successfully.")
        self.show()

    # renders 3D visualization within the vtk_widget
    def render_3d_visualization(self):
        try:
            self.vtk_widget.clear()
            self.log_message("Rendering initial 3D visualization...")
            demo.render_3d(self.vtk_widget,'Totalsegmentator_dataset_v201/s0001/segmentations')
            self.vtk_widget.update()
            self.log_message("Initial 3D visualization rendered successfully.")
        except Exception as e:
            error_message = f"Error rendering 3D visualization: {str(e)}"
            self.log_message(error_message)

    # setting up the menu bar with the actions for the "File", "Edit", "View" and "Help" menus. 
    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        edit_menu = menu_bar.addMenu('Edit')
        view_menu = menu_bar.addMenu('View')
        help_menu = menu_bar.addMenu('Help')

        open_action = QAction('Upload data', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.on_open_file)
        file_menu.addAction(open_action)

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_toggle_action = QAction('Toggle View', self)
        view_toggle_action.setShortcut('Ctrl+T')
        view_toggle_action.triggered.connect(self.on_toggle_view)
        view_menu.addAction(view_toggle_action)

        about_action = QAction('About', self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

    # responding to toggle view action present in the menu bar

    def on_toggle_view(self):
        print("View toggled")

    # opens a file dialog to choose NIfTI file and later on display it 
    def on_open_file(self):
        self.log_message("Open file action triggered")
        directory_name = QFileDialog.getExistingDirectory(self, "QFileDialog")
        if directory_name:
            try:
                file_name = os.path.join(directory_name, 'ct.nii.gz')
                self.log_message(f"Selected file: {file_name}")
                segmentation_dir = os.path.join(directory_name, 'segmentations')
                self.log_message("Converting NIfTI file...")
                image_dim0, image_dim1, image_dim2 = convert_nifti(file_name, logger=self.log_message)
                self.log_message("NIfTI file converted successfully.")

                self.slider.setSliderPosition(0)
                self.image_collection0 = image_dim0
                self.slider.setRange(0, len(self.image_collection0) - 1)
                self.slider1.setSliderPosition(0)
                self.image_collection1 = image_dim1
                self.slider1.setRange(0, len(self.image_collection1) - 1)
                self.slider2.setSliderPosition(0)
                self.image_collection2 = image_dim2
                self.slider2.setRange(0, len(self.image_collection2) - 1)

                self.image_placeholder.setPixmap(self.image_collection0[0])
                self.image_placeholder1.setPixmap(self.image_collection1[0])
                self.image_placeholder2.setPixmap(self.image_collection2[0])

                self.log_message("Rendering 3D visualization...")
                try:
                    self.vtk_widget.clear()
                    demo.render_3d(self.vtk_widget, segmentation_dir)
                    self.vtk_widget.update()
                    self.log_message("3D visualization rendered successfully.")
                except Exception as e:
                    error_message = f"Error rendering 3D visualization: {str(e)}"
                    self.log_message(error_message)
            except Exception as e:
                error_message = f"Error in on_open_file: {str(e)}"
                self.log_message(error_message)

    # dislays the information after clicking the option "Save" file 
    def on_save_file(self):
        self.log_message("Save file action triggered")

    # displays the "About" dialog with application information
    def on_about(self):
        self.log_message("About action triggered")
        QMessageBox.about(self, "About", "SegMed 1.2\nDeveloped by ...")

    # handling the events when the image slider is moved. Updates the displayed image
    def on_slider_move(self):
        self.image_placeholder.setPixmap(self.image_collection0[self.slider.value()])
        self.image_placeholder1.setPixmap(self.image_collection1[self.slider1.value()])
        self.image_placeholder2.setPixmap(self.image_collection2[self.slider2.value()])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = MedicalImageViewer()
    sys.exit(app.exec_())
