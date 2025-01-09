# === GUI.py ===

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import sys
import torch

from matplotlib import cm
from matplotlib.colors import to_rgb
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPixmap, QImage, QDesktopServices
from PyQt5.QtCore import Qt, QSize, QUrl, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
from pyvistaqt import QtInteractor
from stl import mesh
from vedo import load, Plotter

import data_import
import demo
import segmentation
import visualization

from data_import import validate_nifti, load_nifti
from manage_view import OrganSelectionDialog 
from model import get_unet_model 
from visualization import display_single_slice, overlay_slices, convert_seg_slices, reorient_scan


# source for the wigets and libraries used: https://doc.qt.io/qt-6/qtwidgets-module.html ; https://doc.qt.io/qt-6/widget-classes.html


# the organ mapping can be found here: https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file
organ = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}

if not organ:
    raise ValueError("ERROR: organ dictionary is empty. No labels have been created.")

# in this step, we create a mapping for numeric labels and organ names:
lbl_to_organ = organ.copy() # first, we create a copy of the original "organ" dictionary and we map labels to organ names
organ_to_lbl = {v: k for k, v in organ.items()}  # next, we create a reverse mapping - we map organ names to labels    source: https://www.geeksforgeeks.org/python-ways-to-invert-mapping-of-dictionary/


# colors for the above 117 classes. Inspired by Slicer 3D color palette;  source: https://www.slicer.org/wiki/Documentation/4.8/Modules/Colors
organ_color = [
    "#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2",
    "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb",
    "#636363", "#969696", "#bdbdbd", "#d9d9d9", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
    "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252",
    "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a",
    "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6", "#3182bd", "#6baed6",
    "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2", "#31a354", "#74c476",
    "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb", "#636363", "#969696",
    "#bdbdbd", "#d9d9d9", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a",
    "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6", "#1f77b4", "#ff7f0e",
    "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c",
    "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c"
]

# mapping for organ names (class indices) to their corresponding colors:
class_to_color = {
    idx: organ_color[idx % len(organ_color)]
    for idx in range(1, len(organ) + 1)
}

# class responsible for load the technical documentation in PDF format:  source: https://gist.github.com/glowinthedark/d9cd06212d06c6148589772dff448f9b
class PDFViewer(QMainWindow):
    def __init__(self, pdf_url):
        super().__init__()
        self.setWindowTitle("SegMed - Technical documentation")
        self.setGeometry(300, 100, 800, 600)

        # QWebEngineView component to display PDFs  source: https://doc.qt.io/qt-5/qwebengineview.html 
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl(pdf_url))  # URL to PDF file 
        self.setCentralWidget(self.browser)

        self.log_message("Loading technical documentation in PDF format ...")


# class allowing the user to zoom-in on an image in a dialog window by double-clicking on it:
class ZoomCT(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter) # alignment set to center 
        self.setScaledContents(True) # guarantees that the label's content can adjust to the label's dimensions
        self.original_pixmap = None # variable to store the original image (pixmap) displayed in the label

    def mouseDoubleClickEvent(self, event): # method to override
        if self.pixmap(): # if label displays a pixmap:
            self.original_pixmap = self.pixmap() # pixmap is stored/saved here 

            dlg = QDialog(self) # modal dialog window created;  source: https://doc.qt.io/qt-6/qdialog.html
            dlg.setWindowTitle("Zoomed view") 
            dlg.setModal(True) 
            dlg.resize(800, 600) 

            zoomed_label = QLabel(dlg) # zoomed image displayed in the label;  source: https://doc.qt.io/qt-6/qdialog.html
            zoomed_label.setAlignment(Qt.AlignCenter) 
            scaled = self.original_pixmap.scaled(
                dlg.width(), dlg.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation   # original pixmap is scaled to fit the dialog window, while maintaing its aspect ratio; source: https://doc.qt.io/qt-6/qt.html  https://doc.qt.io/qt-6/qgraphicspixmapitem.html
            )
            zoomed_label.setPixmap(scaled)

            layout = QVBoxLayout() # layout created to hold zoomed_label 
            layout.addWidget(zoomed_label) # layout is added to the dialog window
            dlg.setLayout(layout) # layout is set as the dialog window's layout

            dlg.exec_() # dialog window is executed

        super().mouseDoubleClickEvent(event) # event passed to the parent class



# class for the main window of the SegMed's application:
class CTViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting up window's properties for CT scans handling logic, their segmentation and visualization:
        self.affine = None # affine transformation; source: https://mathworld.wolfram.com/AffineTransformation.html
        self.ct_scans = None # loaded CT scans
        self.curr_seg_lbls = [] # currently selected segmented labels (anatomical structures)
        self.init_ui() # UI components initialization
        self.loaded_vol = {} # loaded 3D volumes for visualization
        self.seg_result = None  # segmentation results 
        self.seg_scans = {} # segmented CT scans
        
    # logger messages 
    def log_message(self, message):
        self.error_log.appendPlainText(message)  # message is appended to the error log for debugging/ informational purposes 


    # rendering 3D visualization of segmented data 
    # we'll do this with the use of vtk widget: https://kitware.github.io/vtk-js/docs/concepts_widgets.html
    def render_3D_visualization(self, seg_file=None):
        try:
            # at first, we clear the wiget to prepare for a new rendering:
            self.vtk_widget.clear() 
            self.log_message("Clearing previous visualization. Rendering a new 3D visualization...")

            # next, we initialize a Plotter (VTK plotter) for the rendering; source: https://vtk.org/doc/nightly/html/classvtkPlot.html
            self.plotter = Plotter(qt_widget=self.vtk_widget) 
            self.plotter.background("#F5F5F5")

            if seg_file: # if segmentation file is provided:
                seg_data = nib.load(seg_file).get_fdata()  # then, we load and process segmentation data from the given file
                self.log_message(f"Segmentation data loaded from {seg_file}.") 

                # identifying all unique labels, except for the background (0):
                unique_lbls = np.unique(seg_data)
                unique_lbls = unique_lbls[unique_lbls != 0]
                self.log_message(f"Unique labels in segmentation: {unique_lbls}.")

                # preparing 3D models for each labelled region (organ):
                volume = [] 

                for lbl in unique_lbls:
                    # extrating organ's name for the label. We are going to use the default one in case name cannot be found:
                    organ_name = lbl_to_organ.get(int(lbl), f'label_{int(lbl)}')
                    organ_mask = (seg_data == lbl).astype(np.uint8)  # source: https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.astype.html

                    if np.sum(organ_mask) == 0: # skipping empty mask, with no data in it
                        self.log_message(f"Skipping label {lbl} ({organ_name}) as it provides no data.")
                        continue

                    # generating STL file for the labeled region and load it as a 3D model:
                    seg_path = f'segmented_{organ_name}.stl'
                    demo.convert_to_stl(organ_mask, seg_path)

                    # assigning permanent color to a class, based on its label: 
                    color_hex = class_to_color.get(int(lbl), "#FFFFFF")
                    rgb_color = to_rgb(color_hex)
                    vol = load(seg_path).color(rgb_color)
                    vol.rotate_z(-90)  # model orientation (adjustment);  source: https://docs.pyvista.org/api/core/_autosummary/pyvista.dataset.rotate_z
                    vol.scale([1, 1, -1]) # scaling ; source: https://vedo.embl.es/autodocs/content/vedo/vedo/mesh.html
                    volume.append(vol)

                    # clearning temp STL files after processing stage:
                    if os.path.exists(seg_path):
                        os.remove(seg_path)

                if volume: # if any volumes have been generated:
                    self.plotter.show(volume, axes=1) # rendering 3D visualization (models) in the VTK widget
                    self.log_message("3D visualization rendered successfully.")
                else:
                    self.log_message("No volumes generated for visualization.")
                    QMessageBox.warning(self, "Visualization warning", "No volumes were generated for visualization.")

            else:
                self.log_message("No segmentation file provided. Visualization skipped.")

            # updating VTK widget after rendering process:
            self.plotter.background("#F5F5F5")
            self.vtk_widget.update()

        except Exception as e:
            error_message = f"ERROR rendering 3D visualization: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Visualization error", error_message)



    # User Interface (UI) initialization
    # Setting up the main window, layouts, wigets, styles visible to the end-user: 
    def init_ui(self):

        # main window title, flags, tooltip font: 
        self.setWindowTitle('SegMed 1.1')
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)  # source: https://doc.qt.io/qtforpython-5/overviews/qtwidgets-widgets-windowflags-example.html
        QToolTip.setFont(QFont('SansSerif', 10))  # source: https://doc.qt.io/qt-6/qtooltip.html

        # screen dimensions:
        screen = QApplication.desktop().screenGeometry()   # sourceL https://doc.qt.io/qt-6/qapplication.html
        screen_width = screen.width()
        screen_height = screen.height()

        # styles  source: https://doc.qt.io/qt-6/stylesheet-examples.html
        self.setStyleSheet("""  
            QMainWindow {
                background-color: #FFFFFF;
            }
            QLabel {
                border: 2px solid #A0A0A0;
                border-radius: 10px;
                background-color: #F5F5F5;
                color: #333333;
                font-family: 'Arial';
                font-size: 14px;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                border: 1px solid #B0B0B0;
                background: #D3D3D3;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ADD8E6, stop:1 #87CEFA);
                border: 1px solid #6495ED;
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #87CEFA, stop:1 #4682B4);
            }
            QPlainTextEdit {
                background-color: #FAFAFA;
                color: #333333;
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                border: 1px solid #B0B0B0;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #87CEFA;
                color: #FFFFFF;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-family: 'Arial';
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4682B4;
            }
            QMenuBar {
                background-color: #F5F5F5;
                color: #333333;
            }
            QMenuBar::item {
                background-color: #F5F5F5;
                padding: 5px 15px;
                color: #333333;
            }
            QMenuBar::item:selected {
                background-color: #ADD8E6;
            }
            QMenu {
                background-color: #FFFFFF;
                color: #333333;
            }
            QMenu::item:selected {
                background-color: #87CEFA;
                color: #FFFFFF;
            }
        """)

        # menu bar:
        self.setup_menu_bar()

        # main layout:
        main_layout = QVBoxLayout()   # source: https://doc.qt.io/qt-6/qvboxlayout.html
        main_layout.setAlignment(Qt.AlignCenter)

        # grid layout for image placeholders and sliders:
        grid_layout = QGridLayout()   # source: https://doc.qt.io/qt-6/qgridlayout.html
        grid_layout.setAlignment(Qt.AlignCenter) # source: https://www.geeksforgeeks.org/qt-alignment-in-pyqt5/
        grid_layout.setSpacing(20)  

        self.scan_list_sagittal = []
        self.scan_list_coronal = []
        self.scan_list_axial = []

        # image placeholders with fixed sizes:
        placeholder_style = """ 
            QLabel {
                border: 2px solid #A0A0A0;
                border-radius: 10px;
                background-color: #F5F5F5;
                color: #333333;
                font-family: 'Arial';
                font-size: 14px;
                font-weight: bold;
            }
        """

        # views:
        self.scan_top_left = ZoomCT("Sagittal view", self) # Sagittal view 
        self.scan_top_left.setFrameStyle(QFrame.StyledPanel)
        self.scan_top_left.setAlignment(Qt.AlignCenter)
        self.scan_top_left.setStyleSheet(placeholder_style)
        self.scan_top_left.setFixedSize(400, 300)
        self.scan_top_left.setScaledContents(True)

        self.scan_top_right = ZoomCT("Coronal view", self) # Coronal view
        self.scan_top_right.setFrameStyle(QFrame.StyledPanel)
        self.scan_top_right.setAlignment(Qt.AlignCenter)
        self.scan_top_right.setStyleSheet(placeholder_style)
        self.scan_top_right.setFixedSize(400, 300)
        self.scan_top_right.setScaledContents(True)


        self.scan_bottom_left = ZoomCT("Axial view", self) # Axial view
        self.scan_bottom_left.setFrameStyle(QFrame.StyledPanel)
        self.scan_bottom_left.setAlignment(Qt.AlignCenter)
        self.scan_bottom_left.setStyleSheet(placeholder_style)
        self.scan_bottom_left.setFixedSize(400, 300)
        self.scan_bottom_left.setScaledContents(True)


        # VTK widget for 3D visualization - enhanced size and style:
        self.vtk_widget = QtInteractor(self)  # source: https://qtdocs.pyvista.org/usage.html
        self.vtk_widget.setMinimumSize(400, 300)
        self.vtk_widget.setStyleSheet("""
            QtInteractor {
                background-color: #FFFFFF;
                border: 2px solid #A0A0A0;
                border-radius: 10px;
            }
        """)

        # sliders:
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #B0B0B0;
                background: #D3D3D3;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 #ADD8E6, stop:1 #87CEFA);
                border: 1px solid #6495ED;
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 #87CEFA, stop:1 #4682B4);
            }
        """

        self.slider_sagittal = QSlider(Qt.Horizontal)   # source: https://doc.qt.io/qt-6/qslider.html
        self.slider_sagittal.valueChanged.connect(self.slider_move)
        self.slider_sagittal.setStyleSheet(slider_style)
        self.slider_sagittal.setFixedWidth(400)

        self.slider_coronal = QSlider(Qt.Horizontal)
        self.slider_coronal.valueChanged.connect(self.slider_move)
        self.slider_coronal.setStyleSheet(slider_style)
        self.slider_coronal.setFixedWidth(400)

        self.slider_axial = QSlider(Qt.Horizontal)
        self.slider_axial.valueChanged.connect(self.slider_move)
        self.slider_axial.setStyleSheet(slider_style)
        self.slider_axial.setFixedWidth(400)


        # inner layouts for each view:
        inner_layout_sagittal = QVBoxLayout()  # source: https://doc.qt.io/qt-6/qvboxlayout.html
        inner_layout_sagittal.addWidget(self.scan_top_left)
        inner_layout_sagittal.addWidget(self.slider_sagittal)
        inner_layout_sagittal.setAlignment(Qt.AlignCenter)

        inner_layout_coronal = QVBoxLayout()
        inner_layout_coronal.addWidget(self.scan_top_right)
        inner_layout_coronal.addWidget(self.slider_coronal)
        inner_layout_coronal.setAlignment(Qt.AlignCenter)

        inner_layout_axial = QVBoxLayout()
        inner_layout_axial.addWidget(self.scan_bottom_left)
        inner_layout_axial.addWidget(self.slider_axial)
        inner_layout_axial.setAlignment(Qt.AlignCenter)


        # VTK container: 
        self.vtk_container = QWidget()  # source: https://doc.qt.io/qt-6/qwidget.html
        self.vtk_container.setMinimumSize(400, 300)
        self.vtk_container.setStyleSheet("background-color: transparent;")

        # VTK container grid layout: 
        vtk_container_layout = QGridLayout()  # source: https://doc.qt.io/qt-6/qgridlayout.html
        vtk_container_layout.setContentsMargins(0, 0, 0, 0)
        vtk_container_layout.setSpacing(0)
        self.vtk_container.setLayout(vtk_container_layout)

        vtk_container_layout.addWidget(self.vtk_widget, 0, 0)

        # zoom in/zoom out buttons and their style:
        self.zoom_in_button = QPushButton("+")  # source: https://doc.qt.io/qt-6/qpushbutton.html
        self.zoom_in_button.setFixedSize(30, 30)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedSize(30, 30)
        self.zoom_out_button.clicked.connect(self.zoom_out)

        button_style = """
            QPushButton {
                background-color: rgba(135, 206, 250, 180);  
                color: #FFFFFF;
                border-radius: 15px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 180);
            }
        """
        self.zoom_in_button.setStyleSheet(button_style)
        self.zoom_out_button.setStyleSheet(button_style)

        # layout holding the buttons:
        buttons_layout = QVBoxLayout()  # source: https://doc.qt.io/qt-6/qvboxlayout.html
        buttons_layout.setContentsMargins(5, 5, 5, 5)
        buttons_layout.setSpacing(5)
        buttons_layout.addWidget(self.zoom_in_button)
        buttons_layout.addWidget(self.zoom_out_button)
        buttons_layout.addStretch()
        buttons_layout.setAlignment(Qt.AlignTop | Qt.AlignRight)

        # widget holding the buttons layout:
        buttons_widget = QWidget() # sources: https://doc.qt.io/qt-6/qwidget.html
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setStyleSheet("background-color: transparent;")

        # adding buttons_widget to the grid layout, overlaid on the vtk_widget:
        vtk_container_layout.addWidget(buttons_widget, 0, 0, Qt.AlignTop | Qt.AlignRight)

        # inner layout for 3D view:
        inner_layout_3d = QVBoxLayout() # source: https://doc.qt.io/qt-6/qvboxlayout.html
        inner_layout_3d.addWidget(self.vtk_widget)
        inner_layout_3d.setAlignment(Qt.AlignCenter)

        # adding inner layouts to grid layout with improved spacing:
        grid_layout.addLayout(inner_layout_sagittal, 0, 0)
        grid_layout.addLayout(inner_layout_coronal, 0, 1)
        grid_layout.addLayout(inner_layout_axial, 1, 0)
        grid_layout.addWidget(self.vtk_container, 1, 1)

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # source: https://doc.qt.io/qt-6/qsizepolicy.html

        main_layout.addWidget(grid_widget, alignment=Qt.AlignCenter)

        # error logs:
        self.error_log = QPlainTextEdit()  # source: https://doc.qt.io/qt-6/qplaintextedit.html
        self.error_log.setReadOnly(True)
        self.error_log.setFixedHeight(150)
        self.error_log.appendPlainText("Error log:\n")
        self.error_log.setStyleSheet("""
            QPlainTextEdit {
                background-color: #FAFAFA;
                color: #333333;
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                border: 1px solid #B0B0B0;
                border-radius: 5px;
            }
        """)
        self.error_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        main_layout.addWidget(self.error_log)


        # welcome message at the start of the logger: 
        self.log_message("Welcome to SegMed! Please choose an action from the menu bar above.")


        # progress bar:
        self.progress_bar = QProgressBar(self)  # source: https://doc.qt.io/qt-6/qprogressbar.html
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #B0B0B0;
                border-radius: 5px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #87CEFA;
                width: 10px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # central widget:
        central_widget = QWidget()  # source: https://doc.qt.io/qt-6/qwidget.html
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        try:
            self.log_message("UI has been initialized successfully.")
        except Exception as e:
            error_message = f"ERROR initializing UI: {str(e)}."
            self.log_message(error_message)

        self.render_3D_visualization()

        self.resize(900, 800)  # initial size comprising all widgets
        self.move((screen_width - self.width()) // 2, (screen_height - self.height()) // 2)
        self.show()


    # setting up the menu bar with File, Edit, View, Help, and About menu actions:
    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
        edit_menu = menu_bar.addMenu('Edit')
        help_menu = menu_bar.addMenu('Help')
        about_menu = menu_bar.addMenu('About')

        # file menu actions:
        upload_action = QAction('Upload data', self)
        upload_action.setShortcut('Ctrl+U')
        upload_action.triggered.connect(self.upload_data)
        file_menu.addAction(upload_action)

        save_action = QAction('Save segmentation', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_segmentation)
        file_menu.addAction(save_action)

        close_action = QAction('Close segmentation', self)
        close_action.setShortcut('Ctrl+C')
        close_action.triggered.connect(self.close_segmentation)
        file_menu.addAction(close_action)

        # edit menu actions:
        segment_action = QAction('Segment a CT scan', self)
        segment_action.setShortcut('Ctrl+G')
        segment_action.triggered.connect(self.segment_image)
        segment_action.setEnabled(False)  # initially disabled until data is uploaded
        edit_menu.addAction(segment_action)
        self.segment_action = segment_action  

        # view menu actions:
        manage_view_action = QAction('Manage view', self)
        manage_view_action.setShortcut('Ctrl+M')
        manage_view_action.triggered.connect(self.manage_view)
        view_menu.addAction(manage_view_action)

        calc_volume_action = QAction('Calculate organ volume', self)
        calc_volume_action.setShortcut('Ctrl+V')
        calc_volume_action.triggered.connect(self.calculate_volume)
        view_menu.addAction(calc_volume_action)

        zoom_in_action = QAction('Zoom in', self)
        zoom_in_action.setShortcut('Ctrl+I')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('Zoom out', self)
        zoom_out_action.setShortcut('Ctrl+O')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        # help menu actions:
        help_action = QAction('Get help in using SegMed', self)
        help_action.setShortcut('Ctrl+H')
        help_action.triggered.connect(self.help)
        help_menu.addAction(help_action)

        report_problem_action = QAction('Report a problem', self)
        report_problem_action.setShortcut('Ctrl+R')
        report_problem_action.triggered.connect(self.report_problem)  
        help_menu.addAction(report_problem_action)

        contribute_action = QAction('Contribute to SegMed development', self)
        contribute_action.setShortcut('Ctrl+T')
        contribute_action.triggered.connect(self.contribute)
        help_menu.addAction(contribute_action)

        # about menu actions:
        about_action = QAction('About SegMed', self)
        about_action.setShortcut('Ctrl+A')
        about_action.triggered.connect(self.about)
        about_menu.addAction(about_action)


    # function to enable uploading the CT scans directly to the SegMed's application
    def upload_data(self):
        self.log_message("Upload data action has been triggered.")

        # options for uploading CT scans:
        options = [
            "Upload CT scan for both segmentation and visualization.", 
            "Upload already segmented CT scan for visualization only."
        ]

        choice, ok = QInputDialog.getItem(  # source: https://doc.qt.io/qt-6/qinputdialog.html
            self, 
            "Select upload option",  # - dialog title 
            "Choose an option:",     # - prompt message 
            options,                 # - list of options 
            0,                       # - initial selection (first option)
            False                    # - only single selection
        )

        if ok and choice: # checking if the user has chosen an option "ok" and should male a choice from two possibilites:
            if choice == "Upload CT scan for both segmentation and visualization.":
                self.log_message("User has chosen to upload CT scan for both segmentation and visualization.")
                self.upload_ct_scan()
            
            elif choice == "Upload already segmented CT scan for visualization only.":
                self.log_message("User has chosen to upload already segmented CT scan for visualization only.")
                self.upload_segmented_ct_scan()


    # function to handle uploading CT scans for the scans that haven't been segmented yet
    def upload_ct_scan(self):
        self.log_message("Upload CT scan action has been triggered.")
        try:
            file_dialog = QFileDialog(self)  # source: https://doc.qt.io/qt-6/qfiledialog.html
            file_dialog.setWindowTitle("Please select a CT scan file to upload:")
            file_dialog.setNameFilter("NIfTI files (*.nii *.nii.gz)")  # only NIfTI files can be uploaded
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            if file_dialog.exec_(): # executing created file dialog and check whether the user select a file: 
                chosen_file = file_dialog.selectedFiles()
                if chosen_file:
                    file_path = chosen_file[0] # path to the selected file
                    self.log_message(f"Uploading CT scan from {file_path}...")

                    # checking if the selected file is a valid NIfTI file:
                    if validate_nifti(file_path):
                        ct_data, affine = load_nifti(file_path) # loading NIfTI file and extracting CT scan as well as affine transformation
                        ct_data, affine = reorient_scan(ct_data, affine, desired_ornt=('R', 'A', 'S'), logger=self.log_message) # reorienting the scan to a desired orientation, pleasing to the human-eye:
                        
                        # saving path to the file, ct scans and affine transformation:
                        self.file_path = file_path
                        self.ct_scans = ct_data
                        self.affine = affine
                        self.log_message("CT scan has been validated and successfully uploaded.")

                        # When user, instead of choosing an option "Close" to close the segmentation, chooses "Upload data" instead, we reset colors and segmentation data: 
                        self.seg_sagittal = []
                        self.seg_coronal = []
                        self.seg_axial = []
                        self.seg_scans = {}
                        self.seg_result = None
                    
                        # resetting class-to-color mappings (to default values): 
                        for idx in range(1, len(organ) + 1):
                            class_to_color[idx] = organ_color[idx % len(organ_color)]
                        
                        # converting the uploaded into slices for better visualization (in all three dimensions):
                        self.scan_list_sagittal, self.scan_list_coronal, self.scan_list_axial, self.affine = visualization.convert_img_slices(
                            file_path, target_size=(400, 300, 128), logger=self.log_message)
                        
                        # updating image placeholders with brand new slices:
                        self.update_img_placeholders()
                        self.segment_action.setEnabled(True) # enable segmentation action as the scan has been uploaded to the application successfully
                        
                        # at last, we can render 3D visualization of the uploaded scan
                        self.render_3D_visualization()
                    else:
                        raise ValueError("Selected file is not a valid NIfTI file.")
        except Exception as e:
            error_message = f"ERROR uploading CT scan: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Upload error", error_message)


    # function to handle uploading CT scans for the scans already segmented
    def upload_segmented_ct_scan(self):
        self.log_message("Upload segmented CT scan action has been triggered.")
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Please select a segmented CT scan file to upload:")
            file_dialog.setNameFilter("NIfTI files (*.nii *.nii.gz)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            if file_dialog.exec_():
                chosen_file = file_dialog.selectedFiles()
                if chosen_file:
                    file_path = chosen_file[0]
                    self.log_message(f"Uploading segmented CT scan from {file_path}...")

                    if validate_nifti(file_path):
                        seg_data, affine = load_nifti(file_path)
                        seg_data, affine = reorient_scan(seg_data, affine, desired_ornt=('R', 'A', 'S'), logger=self.log_message)

                        self.seg_result = seg_data
                        self.affine = affine
                        self.log_message("Segmented CT scan has been successfully uploaded.")

                        self.render_3D_visualization_seg(seg_data)

                        # as the segmentation has been already performed, segmentation action is disabled
                        self.segment_action.setEnabled(False)
                    else:
                        raise ValueError("Selected file is not a valid NIfTI file.")

        except Exception as e:
            error_message = f"ERROR uploading segmented CT scan: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Upload error", error_message)

    # function to manage segmentation of the uploaded CT scan: 
    def segment_image(self):
        self.log_message("Segment image action has been triggered.")
        try:
            if self.ct_scans is None: # time to check whether scans have been uploaded before doing segmentation
                self.log_message("No CT scans have been loaded for segmentation.")
                QMessageBox.warning(self, "Segmentation error",
                                    "Please upload your CT scan before performing segmentation.")
                return

            # progress bar has been reset, starting segmentation process:
            self.progress_bar.setValue(0)
            self.log_message("Starting segmentation...")

            # timer will be used to simulate progress updates onthe progress bar:
            def simulate_progress():
                value = self.progress_bar.value()
                if value < 100:
                    value += 5  # incrementing progress bar value by 5% each 
                    self.progress_bar.setValue(value)
                else:
                    timer.stop() # stopping timer when progress bar value reaches 100% 
                    self.log_message("Segmentation completed.")
                    QMessageBox.information(self, "Segmentation complete", "Segmentation has been successfully completed.")
                    self.progress_bar.setValue(0)  # resetting progress bar value to 0

            # starting a timer that updates the progress bar every 100 ms:  (simulation of progress)
            timer = QTimer(self)
            timer.timeout.connect(simulate_progress)
            timer.start(100)  # 100 ms

            # at this point, the segmentation process has started:
            seg_out = segmentation.segment_img_with_TS(self.file_path) 
            seg_data = seg_out.get_fdata()  # extracting segmentation data;  source: https://nipy.org/nibabel/images_and_memory.html
            self.seg_result = seg_data # storing segmentation result

            # converting segmentation data into slices for each axis views: 
            self.seg_sagittal, self.seg_coronal, self.seg_axial = visualization.convert_seg_slices(
                seg_data=seg_data,
                target_size=(400,300,128),
                logger=self.log_message
            )

            # rendering 3D visualization of the segmentation:
            self.render_3D_visualization_seg(seg_out.get_fdata())

            self.log_message("Segmentation completed. You can now save the segmentation from the 'File' menu.")

        except Exception as e:
            error_message = f"ERROR: Error while performing segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Segmentation error", error_message)


    # function for rendering 3D visualization from segmentation data:
    def render_3D_visualization_seg(self, seg_data):
        try:
            # at first, we clear the wiget to prepare for a new rendering (if any exists):
            self.vtk_widget.clear()
            self.log_message("Rendering 3D visualization from segmentation data...")

            # next, we initialize a Plotter (VTK plotter) for the rendering; source: https://vtk.org/doc/nightly/html/classvtkPlot.html
            self.plotter = Plotter(qt_widget=self.vtk_widget)
            self.plotter.background("#F5F5F5") # background color
            self.loaded_vol.clear()

            # checking if segmentation data is valid (not None, not empty):
            if seg_data is None or seg_data.size == 0:
                self.log_message("Segmentation data is empty or None.")
                QMessageBox.warning(self, "Visualization error", "No valid segmentation data provided.")
                return

            # extracting unique labels from segmentation (from 117 possible), excluding background:
            unique_lbls = np.unique(seg_data)
            unique_lbls = unique_lbls[unique_lbls != 0]
            if len(unique_lbls) == 0:
                self.log_message("No valid labels found in segmentation data.")
                QMessageBox.warning(self, "Visualization error", "No valid labels found in segmentation data.")
                return

            # storing current segmentation labels for volume calculation (if such option chosen):
            self.curr_seg_lbls = unique_lbls

            self.log_message(f"Unique labels in segmentation: {unique_lbls}")

            for lbl in unique_lbls:
                organ_name = lbl_to_organ.get(int(lbl), f'label_{int(lbl)}')  # extracting organ name or use default name if not found
                organ_mask = (seg_data == lbl).astype(np.uint8)   # creating mask for each label; source: https://www.programiz.com/python-programming/numpy/methods/astype
                if np.sum(organ_mask) == 0:
                    self.log_message(f"Skipping label {lbl} ({organ_name}), no data found.")
                    continue

                seg_path = f'segmented_{organ_name}.stl'
                demo.convert_to_stl(organ_mask, seg_path)

                # assigning color to the label based on its clss 
                color_hex = class_to_color.get(int(lbl), "#FFFFFF")
                rgb_color = to_rgb(color_hex)

                # loading the STL file into a VTK volume and apply transformations:
                vol = load(seg_path).color(rgb_color)
                # as initial dimensions are +x on the right, +y on the top, +z to the screen , we rotate 90 degrees around x-axis to get the correct, eye-pleasing orientation  
                vol.rotate_x(90)
                # flipping y-axis 
                vol.scale([1, -1, 1])

                # adding the loaded volume with its organ name to the plotter:
                self.loaded_vol[organ_name] = vol

                # removing the temporary STL file:
                if os.path.exists(seg_path):
                    os.remove(seg_path)
                
            # rendering 3D visualization (after checking if any volumes were loaded):
            if self.loaded_vol:
                self.plotter.show(list(self.loaded_vol.values()), axes=1)
                self.log_message("3D visualization rendered successfully.")
            else:
                self.log_message("No volumes were generated for visualization.")
                QMessageBox.warning(self, "Visualization warning", "No volumes were generated for visualization.")

            self.plotter.background("#F5F5F5")
            self.vtk_widget.update()

        except Exception as e:
            error_message = f"ERROR rendering 3D visualization: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Visualization error", error_message)


    # function for updating image placeholders (sagittal, coronal, axial) with the first slice from each view:
    def update_img_placeholders(self):
        if self.scan_list_sagittal:
            visualization.display_single_slice(self.scan_top_left, self.scan_list_sagittal[0])
            self.slider_sagittal.setMaximum(len(self.scan_list_sagittal) - 1) # slider range; source: https://www.geeksforgeeks.org/pyqt5-how-to-set-the-maximum-value-of-progress-bar/ 
            self.slider_sagittal.setValue(0) # reset sider position to the first slice

        if self.scan_list_coronal:
            visualization.display_single_slice(self.scan_top_right, self.scan_list_coronal[0])
            self.slider_coronal.setMaximum(len(self.scan_list_coronal) - 1)
            self.slider_coronal.setValue(0)

        if self.scan_list_axial:
            visualization.display_single_slice(self.scan_bottom_left, self.scan_list_axial[0])
            self.slider_axial.setMaximum(len(self.scan_list_axial) - 1)
            self.slider_axial.setValue(0)

    # function for saving segmented CT scans to a file:
    def save_segmentation(self):
        self.log_message("Save segmentation action has been triggered.")
        try:
            if self.seg_result is not None:  # checking if segmentation data is available to be saved:
                save_path, _ = QFileDialog.getSaveFileName(  # source: https://doc.qt.io/qt-6/qfiledialog.html
                    self,
                    "Save segmentation result", # - dialog title
                    "seg_result.nii.gz",  # - default file name
                    "NIfTI files (*.nii *.nii.gz)"  # - file filter (only NIfTI files)
                )
                if save_path:
                    segmentation.save_segmentation(self.seg_result, self.affine, save_path) # saving the segmentation using provided path
                    self.log_message(f"Segmentation saved at {save_path}.")
                    QMessageBox.information(self, "Save successful", f"Segmentation has been saved at:\n{save_path}")
            else:
                QMessageBox.information(self, "No segmentation", "There is no segmentation data to save.")
        except Exception as e:
            error_message = f"ERROR saving segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Save error", error_message)


    # function for closing the segmentation and resetting the application to its initial state:
    def close_segmentation(self):
        self.log_message("Close segmentation action has been triggered.")
        try:
            # clearing segmentation data (CT scans) and results:
            self.seg_scans = {}
            self.seg_result = None

            # clearing CT scans and affine transformation:
            self.ct_scans = None
            self.affine = None

            # clearing list of segmentation views (sagittal, coronal, axial) -> for 2D segmentation:
            self.seg_sagittal = []
            self.seg_coronal = []
            self.seg_axial = []


            # clearing image placeholders and reseting their labels: 
            self.scan_list_sagittal = []
            self.scan_list_coronal = []
            self.scan_list_axial = []
            
            self.scan_top_left.clear()  # source: https://www.geeksforgeeks.org/pyqt5-how-to-clear-the-content-of-label-clear-and-settext-method/
            self.scan_top_left.setText("Sagittal view")  # source: https://www.geeksforgeeks.org/pyqt5-how-to-change-text-of-pre-existing-label-settext-method/
            self.scan_top_right.clear() # source: https://www.javatpoint.com/python-list-clear-method
            self.scan_top_right.setText("Coronal view")
            self.scan_bottom_left.clear()
            self.scan_bottom_left.setText("Axial view")

            # resetting sliders for each view (to their initial, middle position):
            self.slider_sagittal.setValue(0) # source: https://community.esri.com/t5/geoprocessing-questions/python-script-help-for-row-setvalue-function/td-p/39566
            self.slider_sagittal.setMaximum(0) # source: https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QSpinBox.html
            self.slider_coronal.setValue(0)
            self.slider_coronal.setMaximum(0)
            self.slider_axial.setValue(0)
            self.slider_axial.setMaximum(0)

            # clearing 3D visualization, update VTK vidget:
            if hasattr(self, 'plotter'):
                self.plotter.clear()
                self.vtk_widget.update()

            # disabling segmentation action (as on start, we don't need it as CT scans have not been loaded yet):
            self.segment_action.setEnabled(False)

            self.log_message("Segmentation data and CT scans have been cleared. Application reset to initial state.")

        except Exception as e:
            error_message = f"ERROR closing segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Close error", error_message)


    # function for handling 'Manage view' action from the menu bar:
    def manage_view(self):
        self.log_message("Manage view action has been triggered.")

        # checking if segmentation results are available:
        if self.seg_result is None or not self.seg_result.any():
            QMessageBox.warning(self, "Manage view", "Please perform segmentation first.")
            return

        # extracting unique organ labels from the segmentation result:
        unique_lbls = np.unique(self.seg_result) # source: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        unique_lbls = unique_lbls[unique_lbls != 0]  # not including the background (label 0)
        organ_lbls = [lbl_to_organ.get(int(lbl), f"Organ {int(lbl)}") for lbl in unique_lbls]

        # open dialog for organ selection to display in 3D view:
        dialog = OrganSelectionDialog(organ_lbls, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_organ = dialog.get_selected_organ()
            self.update_3D_view([selected_organ])  # updating 3D view with the selected organ only
        elif dialog.was_closed:
            self.log_message("Dialog was closed. Returning to the full segmentation view.")
            self.update_3D_view(organ_lbls)  # showing all organs


    # function for updating 3D view with selected organs:
    def update_3D_view(self, selected_organs):
        self.log_message(f"Updating view for selected organs: {selected_organs}")

        # adjustng organs visibility:
        for organ_name, vol in self.loaded_vol.items():
            if organ_name in selected_organs:
                vol.alpha(1)  # show organ
            else:
                vol.alpha(0)  # hide organ

        # re-rendering the updated 3D view and refreshing the VTK widget:
        self.plotter.render()  
        self.vtk_widget.update()


    # function for calculating volume of a given organ from the displayed 3D segmentation:
    def calculate_volume(self):
        try:
            # checking if segmentation results and affine transformations are available:
            if self.seg_result is None or self.affine is None:
                QMessageBox.warning(self, "No segmentation",
                                    "No segmentation data found. Please segment or upload a segmented scan first.")
                return

            # checking if we have segmented labels in the current segmentation:
            if self.curr_seg_lbls is None or self.curr_seg_lbls.size == 0:
                QMessageBox.warning(self, "No organs", "No organs found in the current segmentation.")
                return

            # list of organ names from the currently segmented labels:
            organs_present = []
            for lbl in self.curr_seg_lbls:
                lbl_int = int(lbl)
                organ_name = lbl_to_organ.get(lbl_int)
                if organ_name is not None:
                    organs_present.append(organ_name)
                else:
                    self.log_message(f"Warning: label {lbl_int} not found in lbl_to_organ.")

            # checkign if any pf the organs, defined in the dataset, are found in the current segmentation:
            if len(organs_present) == 0:
                QMessageBox.warning(self, "No organs", "No known organs found in the current segmentation.")
                return

            # displaying a dialog for organ selection (to calculate its volume):
            organ_choice, ok = QInputDialog.getItem(  # source: https://doc.qt.io/qt-6/qinputdialog.html
                self,
                "Calculate volume", # - dialog title
                "Choose an organ:", # - prompt message
                organs_present,   # - list of items (organs available)
                0, # - selection index
                False # - single selection
            )

            # calculating volume for the selected organ (if user selects this option):
            if ok and organ_choice:
                try:
                    self.calculate_volume_for_organ(organ_choice)
                except Exception as calc_error:
                    self.log_message(f"ERROR calculating volume: {calc_error}")
                    QMessageBox.critical(self, "Volume calculation error",
                                         f"An error occurred while calculating volume: {calc_error}")

        except Exception as e:
            self.log_message(f"Unexpected error in calculate_volume: {e}")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")


    # function for calculating volume logic of a given organ from the displayed 3D segmentation:
    # source to calculate volume: https://slicer.readthedocs.io/en/v4.11/user_guide/modules/segmentstatistics.html
    # https://neurostars.org/t/calculate-volumes-counting-voxels-is-a-correct-approach/1433

    """ The logic is as follows:
    1) Label identification: each organ or ROI is assigned a unique label.
    2) Voxel counting: we need to determine how many voxels are in each label.
    3) Voxel volume calculation: volume of a single voxel is determined by the determinant of the affine matrix/transformation.
    4) Total volume calculation: we multiply the number of voxels by the volume of each voxel. To convert from cubic millimeters to milliliters, we divide by 1000.
    """

    # function for calculating volume of a given organ from the displayed 3D segmentation:
    def calculate_volume_for_organ(self, organ_name):
        if self.seg_result is None or self.affine is None:
            QMessageBox.warning(self, "No segmentation",
                                "No segmentation or affine data found. Please ensure you have segmented or uploaded a segmented scan.")
            return

        # getting label id for the selected organ from organ_to_lbl dictionary:
        label_id = organ_to_lbl.get(organ_name, None)
        if label_id is None:
            QMessageBox.information(self, "Organ not found", f"No such organ label found for {organ_name}.")
            return

        count_voxels = np.sum(self.seg_result == label_id)
        if count_voxels == 0:
            QMessageBox.information(self, "No volume", f"No voxels found for {organ_name} in the current segmentation.")
            return

        try:
            voxel_vol = np.abs(np.linalg.det(self.affine))
            organ_vol = count_voxels * voxel_vol
            organ_vol_ml = organ_vol / 1000.0

            QMessageBox.information(self, "Organ volume",
                                    f"The volume for {organ_name} is approximately {organ_vol_ml:.2f} ml.")
        except Exception as e:
            self.log_message(f"ERROR calculating volume: {e}")
            QMessageBox.critical(self, "Volume calculation error", f"An error occurred while calculating volume: {e}")


    # function for handling 'Zoom in' action from the menu bar:
    def zoom_in(self):
        self.log_message("Zoom in action has been triggered.")
        try:
            # checking if plotter is available:  source: https://www.w3schools.com/python/ref_func_hasattr.asp
            if hasattr(self, 'plotter'):
                self.plotter.zoom(1.1)  # zoom in by a factor of 1.2
                self.plotter.render() # re-rendering the plot
                self.vtk_widget.update() # updating/refreshing the widget
            else:
                self.log_message("No plotter available for zooming.")
        except Exception as e:
            error_message = f"ERROR during zoom in: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Zoom in error", error_message)


    # function for handling 'Zoom out' action from the menu bar:
    def zoom_out(self):
        self.log_message("Zoom out action has been triggered.")
        try:
            if hasattr(self, 'plotter'):
                self.plotter.zoom(0.9)  # zoom out by a factor of 0.8
                self.plotter.render() # re-rendering the plot
                self.vtk_widget.update() # updating/refreshing the widget
            else:
                self.log_message("No plotter available for zooming.")
        except Exception as e:
            error_message = f"ERROR during zoom out: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Zoom out error", error_message)


    # function for displaying 'Help' message:
    def help(self):
        self.log_message("Help action has been triggered.")
        help_txt = """
        <h3>SegMed 1.1 - User guide</h3>
        <p>1. Select <b>file -> Upload data</b> to upload a CT scan.</p>
        <p>2. Use the <b>edit -> Segment a CT scan</b> option to segment the scan.</p>
        <p>3. In case of ready-to-upload segmentation, use the <b>Upload already segmented CT scan for visualization only</b> option.</p>
        <p>4. Visualize the results and manage the view from within the application.</p>
        <p>5. If needed, use the <b>Help -> Report a problem</b> option to report an issue.</p>
        <p>
            To view the full documentation: 
            <a href='https://drive.google.com/uc?export=download&id=1f4N7go6s8Td242nYuRbqhYa_GSqhcJ7A' target='_blank'>click here</a>.
        </p>
        """
        # displaying the help message; source: https://doc.qt.io/qt-6/qmessagebox.html   https://doc.qt.io/qt-6/qlabel.html
        msg = QMessageBox(self)
        msg.setWindowTitle("Help") 
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_txt)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


    # function for displaying report problem diagnostic information:
    def report_problem(self):
        self.log_message("Report problem action has been triggered.")
        subject = "Problem report - SegMed 1.1"
        body = "Please describe the issue you have encountered:\n\nSteps to reproduce:\n\nAdditional information:"
        email_address = "segmed.help@gmail.com"
        mailto_link = f"mailto:{email_address}?subject={subject}&body={body}"  # create a link to the default mailbox and open in the default email client:
        QDesktopServices.openUrl(QUrl(mailto_link))  # source: https://doc.qt.io/qt-6/qdesktopservices.html


    # function for displaying "Contribute to SegMed development" information:
    def contribute(self):
        self.log_message("Contribute action has been triggered.")
        github_url = "https://github.com/Klossek02/Inzynierka-TotalSegmentator"  # Link to repository
        QDesktopServices.openUrl(QUrl(github_url)) # source: https://doc.qt.io/qt-6/qdesktopservices.html


    # function displaying "About" dialog with application name, its version, and authors:
    def about(self):
        self.log_message("About action has been triggered.")

        dialog = QDialog(self) # source: https://doc.qt.io/qt-6/qdialog.html
        dialog.setWindowTitle("About")
        dialog.setFixedSize(400, 300)

        about_txt = """
        <div style="text-align: center; font-size: 16px;">
            <b><i>SegMed - 3D medical image segmentation</i></b><br><br>
            <b><i>Version 1.1</i></b><br><br>
            Authors: 
            <ul style="list-style-type: disc; padding-left: 20px; text-align: left;">
                <li>Aleksandra Kłos</li>
                <li>Olga Czajkowska</li>
                <li>Magdalena Leymańczyk</li>
            </ul>
        </div>
        """

        layout = QVBoxLayout(dialog)  # source: https://doc.qt.io/qt-6/qvboxlayout.html
        label = QLabel(about_txt) # source: https://doc.qt.io/qt-6/qlabel.html
        label.setTextFormat(Qt.RichText)
        layout.addWidget(label)

        dialog.exec_()

    # function for handling image slider movement for each image placeholder:
    def slider_move(self):
        try:
            # retrieving current slider positions for sagittal, coronal, axial views:
            curr_sagittal = self.slider_sagittal.value()
            curr_coronal = self.slider_coronal.value()
            curr_axial = self.slider_axial.value()

            # updating sagittal view on the current slider position:
            if curr_sagittal < len(self.scan_list_sagittal):
                base_slice = self.scan_list_sagittal[curr_sagittal]
                if hasattr(self, 'seg_sagittal') and self.seg_sagittal and curr_sagittal < len(self.seg_sagittal): # source: https://www.w3schools.com/python/ref_func_hasattr.asp
                    mask_slice = self.seg_sagittal[curr_sagittal]
                    pixmap_overlay = overlay_slices(
                        base_slice=base_slice,
                        mask_slice=mask_slice,
                        alpha=0.3,  # overlay transparency
                        class_to_color=class_to_color  # map class to color
                    )
                    visualization.display_single_slice(self.scan_top_left, pixmap_overlay)
                else:
                    visualization.display_single_slice(self.scan_top_left, base_slice)

            # updating coronal view on the current slider position:
            if curr_coronal < len(self.scan_list_coronal):
                base_slice = self.scan_list_coronal[curr_coronal]
                if hasattr(self, 'seg_coronal') and self.seg_coronal and curr_coronal < len(self.seg_coronal):
                    mask_slice = self.seg_coronal[curr_coronal]
                    pixmap_overlay = overlay_slices(
                        base_slice=base_slice,
                        mask_slice=mask_slice,
                        alpha=0.3,
                        class_to_color=class_to_color
                    )
                    visualization.display_single_slice(self.scan_top_right, pixmap_overlay)
                else:
                    visualization.display_single_slice(self.scan_top_right, base_slice)

            # updating axial view on the current slider position:
            if curr_axial < len(self.scan_list_axial):
                base_slice = self.scan_list_axial[curr_axial]
                if hasattr(self, 'seg_axial') and self.seg_axial and curr_axial < len(self.seg_axial):
                    mask_slice = self.seg_axial[curr_axial]
                    pixmap_overlay = overlay_slices(
                        base_slice=base_slice,
                        mask_slice=mask_slice,
                        alpha=0.3,
                        class_to_color=class_to_color
                    )
                    visualization.display_single_slice(self.scan_bottom_left, pixmap_overlay)
                else:
                    visualization.display_single_slice(self.scan_bottom_left, base_slice)

        except Exception as e:
            error_message = f"ERROR moving sliders: {str(e)}"
            self.log_message(error_message)


# main execution part:
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = CTViewer()
    sys.exit(app.exec_())