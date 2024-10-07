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
        self.setWindowTitle('3d segmentator') #working title
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.resize(810, 700)  # window size
        QToolTip.setFont(QFont('SansSerif', 10))  # font for tooltips

        self.setupMenuBar()  # menu bar
        
        # main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignCenter)

        self.image_collection0 = []
        self.image_collection1 = []
        self.image_collection2 = []
        
        self.image_placeholder = QLabel('Image View')
        self.image_placeholder.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder.setAlignment(Qt.AlignCenter)
        self.image_placeholder.setFixedSize(QSize(400, 300))

        self.image_placeholder1 = QLabel('Image View')
        self.image_placeholder1.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder1.setAlignment(Qt.AlignCenter)
        self.image_placeholder1.setFixedSize(QSize(400, 300))

        self.image_placeholder2 = QLabel('Image View')
        self.image_placeholder2.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder2.setAlignment(Qt.AlignCenter)
        self.image_placeholder2.setFixedSize(QSize(400, 300))

        self.image_placeholder3 = QLabel('model')
        self.image_placeholder3.setFrameStyle(QFrame.StyledPanel)
        self.image_placeholder3.setAlignment(Qt.AlignCenter)
        self.image_placeholder3.setFixedSize(QSize(400, 300))

        self.vtk_widget = QtInteractor(self.image_placeholder3)

        inner_layout = QVBoxLayout()
        inner_layout.addWidget(self.image_placeholder)
        inner1_layout = QVBoxLayout()
        inner1_layout.addWidget(self.image_placeholder1)
        inner2_layout = QVBoxLayout()
        inner2_layout.addWidget(self.image_placeholder2)
        inner3_layout = QVBoxLayout()
        inner3_layout.addWidget(self.vtk_widget)

        self.plotter = demo.render_3d(self.vtk_widget, 'C:/Users/Dell/Downloads/Totalsegmentator_dataset_v201/s0001/segmentations') #path to some example segmentations file, the entire command can be removed if needed
        self.plotter.show()

        self.image_collection0.append(QPixmap('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/example images/Dim0_Slice128.png'))#initial image, can be anything
        self.image_placeholder.setPixmap(self.image_collection0[0])
        self.image_collection1.append(QPixmap(
            'C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/example images/Dim1_Slice128.png'))
        self.image_placeholder1.setPixmap(self.image_collection1[0])
        self.image_collection2.append(QPixmap(
            'C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/example images/Dim2_Slice128.png'))
        self.image_placeholder2.setPixmap(self.image_collection2[0])

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFixedSize(QSize(400, 20))
        self.slider.valueChanged.connect(self.on_slider_move)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setFixedSize(QSize(400, 20))
        self.slider1.valueChanged.connect(self.on_slider_move)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setFixedSize(QSize(400, 20))
        self.slider2.valueChanged.connect(self.on_slider_move)

        inner_layout.addWidget(self.slider)
        inner1_layout.addWidget(self.slider1)
        inner2_layout.addWidget(self.slider2)

        grid_layout.addLayout(inner_layout, 0, 0)
        grid_layout.addLayout(inner1_layout, 0, 1)
        grid_layout.addLayout(inner2_layout, 1, 0)
        grid_layout.addLayout(inner3_layout, 1, 1)

        main_layout.addLayout(grid_layout)
        
        # main layout to central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.show()

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
        cv2.imwrite('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/Dim0_Slice.png', outputArray)
        pixmap = QPixmap('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/Dim0_Slice.png')
        outputArray0.append(pixmap)

    for i in range(scanArrayShape[1]):
        outputArray = cv2.resize(scanArray[:, i, :], (newScanDims[2], newScanDims[0]))
        cv2.imwrite('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/Dim0_Slice.png', outputArray)
        pixmap = QPixmap('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/Dim0_Slice.png')
        outputArray1.append(pixmap)

    for i in range(scanArrayShape[2]):
        outputArray = cv2.resize(scanArray[:, :, i], (newScanDims[1], newScanDims[0]))
        cv2.imwrite('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/Dim0_Slice.png', outputArray)
        pixmap = QPixmap('C:/Users/Dell/Desktop/studia/6/Artificial Intelligence Fundamentals/Dim0_Slice.png')
        outputArray2.append(pixmap)

    return outputArray0, outputArray1, outputArray2


# def transform3d(file_name):
#     model = get_unet_model(num_classes=2, in_channels=2)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.load_state_dict(torch.load("best_metric_model.pth"))
#
#     img = nib.load(file_name).get_fdata()
#     resized_img = custom_resize(img, spatial_size=(128, 128, 128))
#
#     # ensuring tensor is of correct type and dimensions
#     resized_img = torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0).to(device)  # adding batch dimension
#
#     # ensuring the resized image is of the correct shape
#     if resized_img.ndim == 5:  # expected shape [batch_size, channels, depth, height, width]
#         outputs = model(resized_img)
#         save_nifti(torch.argmax(outputs, dim=1), 'segmentation', 0)
#     else:
#         print(f"Skipping image {file_name} due to incorrect dimensions after resizing: {resized_img.shape}")
#
#     # URL of the SlicerWeb server
#     url = 'http://localhost:2016/slicer/mrml/file'
#     print('slicer found')
#
#     load_data = {
#         "localfile": file_name,
#         "filetype": "VolumeFile"
#     }
#
#     response = requests.post(url, params=load_data)
#     print(response.json())
#
#     url = 'http://localhost:2016/slicer/gui'
#     response = requests.put(url, params={"contents": "viewers", "viewersLayout": "oneup3d"})
#     print(response.json())
#     url = 'http://localhost:2016/slicer/exec'
#     command = "exec(open('C:/Users/Dell/Downloads/slicerScript.py').read())"
#     response = requests.get(url, params={"source": command})
#     print(response.json())

# def custom_resize(img, spatial_size=(128, 128, 128)):
#     transforms = Compose([
#         lambda x: np.expand_dims(x, axis=0),  # adding channel dimension
#         Resize(spatial_size=spatial_size)
#     ])
#     img = transforms(img)
#
#     # adding a second channel if the model expects 2 channels
#     if img.shape[0] == 1:
#         img = np.repeat(img, 2, axis=0)
#
#     return img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = MedicalImageViewer()
    sys.exit(app.exec_())