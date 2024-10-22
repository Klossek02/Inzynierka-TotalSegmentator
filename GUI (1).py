import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor
import os
import requests
#from model import get_unet_model
import numpy as np
import nibabel as nib
import torch
from monai.transforms import Compose, Resize
import demo


# function created to save predicted volume as NIftI file
def save_nifti(volume, path, index=0):
    volume = np.array(volume.detach().cpu()[0], dtype=np.float32)
    volume = nib.Nifti1Image(volume, np.eye(4))
    nib.save(volume, os.path.join(path, f'patient_predicted_{index}.nii.gz'))
    print(f'patient_predicted_{index} is saved', end='\r')

class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SegMed 1.2')
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        QToolTip.setFont(QFont('SansSerif', 10))  # Font for tooltips

        # Get the screen geometry
        screen = QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Set the window's maximum size to the screen size
        self.setMaximumSize(screen_width, screen_height)

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

        self.setupMenuBar()  # Menu bar

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # Grid layout for the visualizations
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignCenter)

        self.image_collection0 = []
        self.image_collection1 = []
        self.image_collection2 = []

        # Image placeholders with fixed size policies
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

        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.vtk_widget.setFixedSize(400, 300)

        # Sliders
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

        # Inner layouts for each visualization and slider
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

        # Add inner layouts to grid layout
        grid_layout.addLayout(inner_layout, 0, 0)
        grid_layout.addLayout(inner1_layout, 0, 1)
        grid_layout.addLayout(inner2_layout, 1, 0)
        grid_layout.addLayout(inner3_layout, 1, 1)

        # Wrap the grid layout into a QWidget
        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        grid_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        grid_widget.adjustSize()  # Adjust size to fit content

        # Add the grid widget to the main layout
        main_layout.addWidget(grid_widget, alignment=Qt.AlignCenter)

        # Add error log panel
        self.error_log = QPlainTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setFixedHeight(150)
        self.error_log.appendPlainText("Error Log:\n")
        self.error_log.appendPlainText("Sample error message 1\nSample error message 2")
        self.error_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        main_layout.addWidget(self.error_log)

        # Set main layout to central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Load initial images
        self.image_collection0.append(QPixmap('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg'))
        self.image_placeholder.setPixmap(self.image_collection0[0])
        self.image_collection1.append(QPixmap('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg'))
        self.image_placeholder1.setPixmap(self.image_collection1[0])
        self.image_collection2.append(QPixmap('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg'))
        self.image_placeholder2.setPixmap(self.image_collection2[0])

        ## Render 3D visualization within the vtk_widget
        self.render_3d_visualization()

        # Adjust the window size based on content
        self.adjustSize()

        # Ensure the window does not exceed the screen size
        window_size = self.size()
        if window_size.width() > screen_width or window_size.height() > screen_height:
            self.resize(screen_width, screen_height)

        # Center the window on the screen
        self.move((screen_width - self.width()) // 2, (screen_height - self.height()) // 2)

        self.show()

    def render_3d_visualization(self):
        # Use the vtk_widget (QtInteractor) to render the 3D visualization
        # Assuming 'demo.render_3d' sets up the plotter with the vtk_widget
        try:
            # Clear any existing plots
            self.vtk_widget.clear()

            # Call your custom render function
            demo.render_3d(self.vtk_widget,
                           'C:/Users/magda/Desktop/Studia/INZYNIERKA/Totalsegmentator_dataset_v201/s0001/segmentations')

            # Render the widget
            self.vtk_widget.update()
        except Exception as e:
            error_message = f"Error rendering 3D visualization: {str(e)}"
            print(error_message)
            self.error_log.appendPlainText(error_message)


    def setupMenuBar(self):
        # menu bar
        menuBar = self.menuBar()

        # top-level menus
        fileMenu = menuBar.addMenu('File')
        editMenu = menuBar.addMenu('Edit')
        viewMenu = menuBar.addMenu('View')
        helpMenu = menuBar.addMenu('Help')

        # actions for file menu
        openAction = QAction('Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.triggered.connect(self.on_open_file)
        fileMenu.addAction(openAction)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # actions for view menu
        viewToggleAction = QAction('Toggle View', self)
        viewToggleAction.setShortcut('Ctrl+T')
        viewToggleAction.triggered.connect(self.on_toggle_view)
        viewMenu.addAction(viewToggleAction)

        # actions for help menu
        aboutAction = QAction('About', self)
        aboutAction.triggered.connect(self.on_about)
        helpMenu.addAction(aboutAction)

    def on_toggle_view(self):
        print("View toggled") # logic to toggle the view

    def on_open_file(self):
        print("Open file action triggered") # logic for opening a file
        #fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  #"All Files (*);;Python Files (*.py)")
        directoryName = QFileDialog.getExistingDirectory(self, "QFileDialog")
        if directoryName:
            fileName = directoryName + '/ct.nii.gz'
            print(fileName)
            directoryName = directoryName + '/segmentations'
            #transform3d(fileName)
            #self.image_placeholder3 = demo.render_3d(self.image_placeholder3)
            imageDim0, imageDim1, imageDim2 = convert_nifti(fileName)
            self.slider.setSliderPosition(0)
            self.image_collection0 = imageDim0
            self.slider.setRange(0, self.image_collection0.__len__()-1)
            self.slider1.setSliderPosition(0)
            self.image_collection1 = imageDim1
            self.slider1.setRange(0, self.image_collection1.__len__()-1)
            self.slider2.setSliderPosition(0)
            self.image_collection2 = imageDim2
            self.slider2.setRange(0, self.image_collection2.__len__()-1)

            self.image_placeholder.setPixmap(self.image_collection0[0])
            self.image_placeholder1.setPixmap(self.image_collection1[0])
            self.image_placeholder2.setPixmap(self.image_collection2[0])

            self.plotter = demo.render_3d(self.vtk_widget, directoryName)
            self.plotter.show()


    def on_save_file(self):
        print("Save file action triggered") # logic for saving a file


    def on_about(self):
        print("About action triggered")

    def on_slider_move(self):
        self.image_placeholder.setPixmap(self.image_collection0[self.slider.value()])
        self.image_placeholder1.setPixmap(self.image_collection1[self.slider1.value()])
        self.image_placeholder2.setPixmap(self.image_collection2[self.slider2.value()])



def convert_nifti(fileName):
    # Load the scan and extract data using nibabel
    scan = nib.load(fileName)
    scanArray = scan.get_fdata()

    # Get and print the scan's shape
    scanArrayShape = scanArray.shape

    # Examine scan's shape and header
    scanHeader = scan.header

    # Calculate proper aspect ratios
    pixDim = scanHeader['pixdim'][1:4]

    # Calculate new image dimensions from aspect ratio
    newScanDims = np.multiply(scanArrayShape, pixDim)
    newScanDims = (round(newScanDims[0]), round(newScanDims[1]), round(newScanDims[2]))

    outputArray0 = []
    outputArray1 = []
    outputArray2 = []

    for i in range(scanArrayShape[0]):
        # Resample the slice
        outputArray = cv2.resize(scanArray[i, :, :], (newScanDims[2], newScanDims[1]))
        # Save the slice as .png image
        cv2.imwrite('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg', outputArray)
        pixmap = QPixmap('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg')
        outputArray0.append(pixmap)

    for i in range(scanArrayShape[1]):
        outputArray = cv2.resize(scanArray[:, i, :], (newScanDims[2], newScanDims[0]))
        cv2.imwrite('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg', outputArray)
        pixmap = QPixmap('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg')
        outputArray1.append(pixmap)

    for i in range(scanArrayShape[2]):
        outputArray = cv2.resize(scanArray[:, :, i], (newScanDims[1], newScanDims[0]))
        cv2.imwrite('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg', outputArray)
        pixmap = QPixmap('C:/Users/magda/Desktop/Studia/INZYNIERKA/ex_slice_1.jpeg')
        outputArray2.append(pixmap)

    return outputArray0, outputArray1, outputArray2


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = MedicalImageViewer()
    sys.exit(app.exec_())
