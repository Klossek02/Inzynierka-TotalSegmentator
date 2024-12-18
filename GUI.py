# === GUI.py ===

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from stl import mesh 
from vedo import load, Plotter
from matplotlib import cm 
from matplotlib.colors import to_rgb
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QSize
from pyvistaqt import QtInteractor

import demo
import data_import
import visualization
import segmentation
from manage_view import ManageViewWindow
from model import get_unet_model  

# wigets and libraries used: https://doc.qt.io/qt-6/qtwidgets-module.html   https://doc.qt.io/qt-6/widget-classes.html


# as in case of the Dataloader, the mapping can be found here: https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file

main_classes_CT = {
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

if not main_classes_CT:
    raise ValueError("ERROR: main_classes_CT dictionary is empty. No labels have been created.")

# in this step, we create a mapping of numeric labels to organ names, based on their grouping in main_classes_CT.
lbl_to_organ = {}
lbl_counter = 1  # let us assume that counting starts from 1.

for organ_grp, organ_names in main_classes_CT.items():
    for organ_name in organ_names:
        lbl_to_organ[lbl_counter] = organ_name
        lbl_counter += 1


SLICER_COLORS = [
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

class_to_color = {
    idx: SLICER_COLORS[idx % len(SLICER_COLORS)]
    for idx in range(1, len(main_classes_CT) + 1)
}



class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ct_scans = None
        self.affine = None
        #self.segmented_scans = {}
        self.organs = []
        self.chosen_organs = []
        self.segmentation_result = None  # Stores the full segmentation result
        self.init_ui()

    def log_message(self, message):
        self.error_log.appendPlainText(message)

    # rendering 3D visualization. 
    # we'll do this with the use of vtk widget: https://kitware.github.io/vtk-js/docs/concepts_widgets.html
    def render_3d_visualization(self, seg_file=None):
        try:
            self.vtk_widget.clear() # at first, we clear the wiget to prepare for a new rendering. 
            self.log_message("Clearning time. Rendering a 3D visualization...")

            self.plotter = Plotter(qt_widget=self.vtk_widget)
            self.plotter.background("#F5F5F5")

            if seg_file:
                seg_data = nib.load(seg_file).get_fdata() # then, we load and process segmentation data. 
                self.log_message(f"Segmentation data has been loaded from {seg_file}")

                # extracting all unique labels, except for the background (0).
                unique_lbls = np.unique(seg_data)
                unique_lbls = unique_lbls[unique_lbls != 0]  
                self.log_message(f"Unique labels in segmentation: {unique_lbls}")

                # preparing 3D models for each labelled region.
                volume = []


                for i, lbl in enumerate(unique_lbls):
                    organ_name = lbl_to_organ.get(int(lbl), f'label_{int(lbl)}')
                    organ_mask = (seg_data == lbl).astype(np.uint8)

                    if np.sum(organ_mask) == 0: # skipping empty mask.
                        self.log_message(f'Skipping label {lbl}.')
                        continue

                    # generating STL file and load it as a 3D model.
                    seg_path = f'segmented_{organ_name}.stl'
                    demo.convert_to_stl(organ_mask, seg_path)
                    print(f'saved {organ_name} as stl')
                    self.organs.append(organ_name)

                    # permanent color for class
                    color_hex = class_to_color.get(int(lbl), "#FFFFFF")  # at default white
                    rgb_color = to_rgb(color_hex)
                    vol = load(seg_path).color(rgb_color)
                    volume.append(vol)

                    # clearning temporary STL files.
                    #if os.path.exists(seg_path):
                    #    os.remove(seg_path)

                # displaying 3D visualization using afornemtioned vtk_widget library.
                self.chosen_organs = self.organs
                self.plotter = Plotter(qt_widget=self.vtk_widget)
                self.plotter.show(volume, axes=1)
                self.log_message("3D visualization has been rendered successfully.")
            else:
                self.log_message("No segmentation file provided, visualization skipped.")

            self.plotter.background("#F5F5F5")
            self.vtk_widget.update()
        except Exception as e:
            error_message = f"Error rendering 3D visualization: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Visualization error", error_message)



    # UI components, layout, styling visible for the user.
    def init_ui(self):
        self.setWindowTitle('SegMed 1.1')
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        QToolTip.setFont(QFont('SansSerif', 10))

        # screen dimensions.
        screen = QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # applying various styles to enhance UI appearance.
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;  // background color
            }
            QLabel {
                border: 2px solid #A0A0A0;
                border-radius: 10px;
                background-color: #F5F5F5;  // label background color
                color: #333333;             // dark text color
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
                background-color: #FAFAFA;  // bright background for log panel 
                color: #333333;             // dark text color 
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                border: 1px solid #B0B0B0;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #87CEFA;  // bright color for buttons  
                color: #FFFFFF;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-family: 'Arial';
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4682B4;  // slightly darker color on a hover 
            }
            QMenuBar {
                background-color: #F5F5F5;  // bright color of the menu bar
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

        self.setup_menu_bar()

        # main layout. 
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        # grid layout - for image placeholders and sliders.
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignCenter)
        grid_layout.setSpacing(20)  

        self.scan_list_sagittal = []
        self.scan_list_coronal = []
        self.scan_list_axial = []

        # image placeholders with fixed sizes.
        placeholder_style = """ 
            QLabel {
                border: 2px solid #A0A0A0;
                border-radius: 10px;
                background-color: #F5F5F5;  // bright background for labels
                color: #333333;
                font-family: 'Arial';
                font-size: 14px;
                font-weight: bold;
            }
        """

        # views
        self.scan_top_left = QLabel("Sagittal view")
        self.scan_top_left.setFrameStyle(QFrame.StyledPanel)
        self.scan_top_left.setAlignment(Qt.AlignCenter)
        self.scan_top_left.setStyleSheet(placeholder_style)
        self.scan_top_left.setFixedSize(400, 300)
        self.scan_top_left.setScaledContents(True)

        self.scan_top_right = QLabel("Coronal view")
        self.scan_top_right.setFrameStyle(QFrame.StyledPanel)
        self.scan_top_right.setAlignment(Qt.AlignCenter)
        self.scan_top_right.setStyleSheet(placeholder_style)
        self.scan_top_right.setFixedSize(400, 300)
        self.scan_top_right.setScaledContents(True)

        self.scan_bottom_left = QLabel("Axial view")
        self.scan_bottom_left.setFrameStyle(QFrame.StyledPanel)
        self.scan_bottom_left.setAlignment(Qt.AlignCenter)
        self.scan_bottom_left.setStyleSheet(placeholder_style)
        self.scan_bottom_left.setFixedSize(400, 300)
        self.scan_bottom_left.setScaledContents(True)

        # VTK widget - enhanced size and style.
        self.vtk_widget = QtInteractor(self)
        self.vtk_widget.setMinimumSize(400, 300)
        self.vtk_widget.setStyleSheet("""
            QtInteractor {
                background-color: #FFFFFF;  
                border: 2px solid #A0A0A0;
                border-radius: 10px;
            }
        """)

        # sliders
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

        self.slider_sagittal = QSlider(Qt.Horizontal)
        self.slider_sagittal.valueChanged.connect(self.on_slider_move)
        self.slider_sagittal.setStyleSheet(slider_style)
        self.slider_sagittal.setFixedWidth(400)
        self.slider_sagittal.setToolTip("Adjust Sagittal Slice")

        self.slider_coronal = QSlider(Qt.Horizontal)
        self.slider_coronal.valueChanged.connect(self.on_slider_move)
        self.slider_coronal.setStyleSheet(slider_style)
        self.slider_coronal.setFixedWidth(400)
        self.slider_coronal.setToolTip("Adjust Coronal Slice")

        self.slider_axial = QSlider(Qt.Horizontal)
        self.slider_axial.valueChanged.connect(self.on_slider_move)
        self.slider_axial.setStyleSheet(slider_style)
        self.slider_axial.setFixedWidth(400)
        self.slider_axial.setToolTip("Adjust Axial Slice")

        # inner layouts for each view.
        inner_layout_sagittal = QVBoxLayout()
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

        self.vtk_container = QWidget()
        self.vtk_container.setMinimumSize(400, 300)
        self.vtk_container.setStyleSheet("background-color: transparent;")

        # grid layout for the container.
        vtk_container_layout = QGridLayout()
        vtk_container_layout.setContentsMargins(0, 0, 0, 0)
        vtk_container_layout.setSpacing(0)
        self.vtk_container.setLayout(vtk_container_layout)

        vtk_container_layout.addWidget(self.vtk_widget, 0, 0)

        # zoom in/zoom out buttons and its style.
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedSize(30, 30)
        self.zoom_in_button.clicked.connect(self.on_zoom_in)
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedSize(30, 30)
        self.zoom_out_button.clicked.connect(self.on_zoom_out)

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

        # layout holding the buttons.
        buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(5, 5, 5, 5)
        buttons_layout.setSpacing(5)
        buttons_layout.addWidget(self.zoom_in_button)
        buttons_layout.addWidget(self.zoom_out_button)
        buttons_layout.addStretch()
        buttons_layout.setAlignment(Qt.AlignTop | Qt.AlignRight)

        # widget holding the buttons layout.
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setStyleSheet("background-color: transparent;")

        # adding buttons_widget to the grid layout, overlaid on the vtk_widget.
        vtk_container_layout.addWidget(buttons_widget, 0, 0, Qt.AlignTop | Qt.AlignRight)

        inner_layout_3d = QVBoxLayout()
        inner_layout_3d.addWidget(self.vtk_widget)
        inner_layout_3d.setAlignment(Qt.AlignCenter)

        # inner layouts to grid layout with improved spacing.
        grid_layout.addLayout(inner_layout_sagittal, 0, 0)
        grid_layout.addLayout(inner_layout_coronal, 0, 1)
        grid_layout.addLayout(inner_layout_axial, 1, 0)
        grid_layout.addLayout(inner_layout_3d, 1, 1)

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout.addWidget(grid_widget, alignment=Qt.AlignCenter)

        # error logs
        self.error_log = QPlainTextEdit()
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

        # central widget
        main_layout.addWidget(self.error_log)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        try:
            self.log_message("UI has been initialized successfully.")
        except Exception as e:
            error_message = f"Error initializing UI: {str(e)}"
            self.log_message(error_message)

        self.render_3d_visualization()

        self.resize(900, 800)  # initial size comprising all widgets.
        self.move((screen_width - self.width()) // 2, (screen_height - self.height()) // 2)
        self.show()


    # setting up the menu bar with File, Edit, View and Help menu actions. 
    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
        edit_menu = menu_bar.addMenu('Edit')
        help_menu = menu_bar.addMenu('Help')
        about_menu = menu_bar.addMenu('About')

        upload_action = QAction('Upload data', self)
        upload_action.setShortcut('Ctrl+U')
        upload_action.triggered.connect(self.on_upload_data)
        file_menu.addAction(upload_action)

        save_action = QAction('Save segmentation', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.on_save_segmentation)
        file_menu.addAction(save_action)

        close_action = QAction('Close segmentation', self)
        close_action.setShortcut('Ctrl+C')
        close_action.triggered.connect(self.on_close_segmentation)
        file_menu.addAction(close_action)

        segment_action = QAction('Segment a CT scan', self)
        segment_action.setShortcut('Ctrl+G')
        segment_action.triggered.connect(self.on_segment_image)
        segment_action.setEnabled(False)  # initially disabled until data is uploaded.
        edit_menu.addAction(segment_action)
        self.segment_action = segment_action  

        manage_view_action = QAction('Manage view', self)
        manage_view_action.setShortcut('Ctrl+M')
        manage_view_action.triggered.connect(self.on_manage_view)
        view_menu.addAction(manage_view_action)

        zoom_in_action = QAction('Zoom in', self)
        zoom_in_action.setShortcut('Ctrl+I')
        zoom_in_action.triggered.connect(self.on_zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('Zoom out', self)
        zoom_out_action.setShortcut('Ctrl+O')
        zoom_out_action.triggered.connect(self.on_zoom_out)
        view_menu.addAction(zoom_out_action)

        help_action = QAction('Get help in using SegMed', self)
        help_action.setShortcut('Ctrl+H')
        help_action.triggered.connect(self.on_help)
        help_menu.addAction(help_action)

        report_problem_action = QAction('Report a problem', self)
        report_problem_action.setShortcut('Ctrl+R')
        report_problem_action.triggered.connect(self.on_report_problem)  
        help_menu.addAction(report_problem_action)

        about_action = QAction('About SegMed', self)
        about_action.setShortcut('Ctrl+A')
        about_action.triggered.connect(self.on_about)
        about_menu.addAction(about_action)


    # function to enable uploading the CT scans directly to the application.
    def on_upload_data(self):
        self.log_message("Upload data action has been triggered.")
        options = [
            "Upload CT scan for both segmentation and visualization.", 
            "Upload already segmented CT scan for visualization only."
        ]

        choice, ok = QInputDialog.getItem(
            self, 
            "Select Upload Option", 
            "Choose an option:", 
            options, 
            0, 
            False
        )

        if ok and choice:
            if choice == "Upload CT scan for both segmentation and visualization.":
                self.log_message("User has chosen to upload CT scan for both segmentation and visualization.")
                self.on_upload_ct_scan()
            
            elif choice == "Upload already segmented CT scan for visualization only.":
                self.log_message("User has chosen to upload already segmented CT scan for visualization only.")
                self.on_upload_segmented_ct_scan()


    def on_upload_ct_scan(self):
        self.log_message("Upload CT scan action has been triggered.")
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Select a CT scan file to upload:")
            file_dialog.setNameFilter("NIfTI Files (*.nii *.nii.gz)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            if file_dialog.exec_():
                chosen_file = file_dialog.selectedFiles()
                if chosen_file:
                    file_path = chosen_file[0]
                    self.log_message(f"Uploading CT scan from {file_path}...")

                    if data_import.validate_nifti(file_path):
                        ct_data, affine = data_import.load_nifti(file_path)
                        self.ct_scans = ct_data
                        self.affine = affine
                        self.log_message("CT scan has been validated and successfully uploaded.")
                        
                        # converting image slices (in all dimensions) for visualization.
                        self.scan_list_sagittal, self.scan_list_coronal, self.scan_list_axial, self.affine = visualization.convert_img_slices(
                            file_path, target_size=(400, 300, 128), logger=self.log_message)
                        
                        self.update_image_placeholders()
                        self.segment_action.setEnabled(True)
                        
                        self.render_3d_visualization()
                    else:
                        raise ValueError("Selected file is not a valid NIfTI file.")
        except Exception as e:
            error_message = f"Error uploading CT scan: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Upload error", error_message)

    def on_upload_segmented_ct_scan(self):
        self.log_message("Upload segmented CT scan action has been triggered.")
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Select a segmented CT scan file to upload:")
            file_dialog.setNameFilter("NIfTI Files (*.nii *.nii.gz)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            if file_dialog.exec_():
                chosen_file = file_dialog.selectedFiles()
                if chosen_file:
                    file_path = chosen_file[0]
                    self.log_message(f"Uploading segmented CT scan from {file_path}...")

                    if data_import.validate_nifti(file_path):
                        seg_data, affine = data_import.load_nifti(file_path)
                        self.segmentation_result = seg_data
                        self.affine = affine
                        self.log_message("Segmented CT scan has been successfully uploaded.")

                        # rendering the segmentation
                        self.render_3d_visualization_from_data(seg_data)

                        # disabling segmentation action, as "Segment image" option has been already executed.
                        self.segment_action.setEnabled(False)
                    else:
                        raise ValueError("Selected file is not a valid NIfTI file.")

        except Exception as e:
            error_message = f"Error uploading segmented CT scan: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Upload error", error_message)

    def on_segment_image(self):
        self.log_message("Segment image action has been triggered.")
        try:
            if self.ct_scans is None:
                self.log_message("No CT scans have been loaded for segmentation.")
                QMessageBox.warning(self, "Segmentation error", "Please upload your CT scan before performing segmentation.")
                return

            # for debugging purposes - to check whether the tensor shape is correct.
            self.log_message("Preprocessing the CT image for segmentation...")
            img_tensor = segmentation.preprocess_img(self.ct_scans, target_size=(128, 128, 128))
            self.log_message(f"Preprocessed image tensor shape: {img_tensor.shape}")  

            self.log_message("Loading the segmentation model...")
            model = self.load_segmentation_model()

            self.log_message("Performing segmentation...")
            seg_out = segmentation.segment_img(model, img_tensor)
            self.log_message(f"Segmentation output shape: {seg_out.shape}")

            # storing segmentation results.
            self.segmentation_result = seg_out

            # we can optionally render the segmentation as well.
            self.render_3d_visualization_from_data(seg_out)

            self.log_message("Segmentation completed. You can now save the segmentation from the 'File' menu.")

        except Exception as e:
            error_message = f"ERROR: error while performing segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Segmentation error", error_message)



    def render_3d_visualization_from_data(self, seg_data):
        try:
            self.vtk_widget.clear()
            self.log_message("Rendering 3D visualization from segmentation data...")

            self.plotter = Plotter(qt_widget=self.vtk_widget)
            self.plotter.background("#F5F5F5")

            # checking if the data are valid
            if seg_data is None or seg_data.size == 0:
                self.log_message("Segmentation data is empty or None.")
                QMessageBox.warning(self, "Visualization Error", "No valid segmentation data provided.")
                return

            # getting unique labels
            unique_lbls = np.unique(seg_data)
            unique_lbls = unique_lbls[unique_lbls != 0]  # skipping background (label 0)
            if len(unique_lbls) == 0:
                self.log_message("No valid labels found in segmentation data.")
                QMessageBox.warning(self, "Visualization Error", "No valid labels found in segmentation data.")
                return

            self.log_message(f"Unique labels in segmentation: {unique_lbls}")

            volume = []

            for lbl in unique_lbls:
            # getting organ name based on the label
                organ_name = lbl_to_organ.get(int(lbl), f'label_{int(lbl)}')

            # creating organ mask for a given label
                organ_mask = (seg_data == lbl).astype(np.uint8)
                if np.sum(organ_mask) == 0:
                    self.log_message(f"Skipping label {lbl}, no data found.")
                    continue

                # generating STL file
                seg_path = f'segmented_{organ_name}.stl'
                try:
                    demo.convert_to_stl(organ_mask, seg_path)
                except Exception as stl_error:
                    self.log_message(f"Error generating STL for {organ_name}: {str(stl_error)}")
                    continue

                # getting color for a given label
                color_hex = class_to_color.get(int(lbl), "#FFFFFF")  # Domyślnie biały
                rgb_color = to_rgb(color_hex)

            # loading STL file as a 3D model
                try:
                    vol = load(seg_path).color(rgb_color)
                    volume.append(vol)
                except Exception as load_error:
                    self.log_message(f"Error loading STL for {organ_name}: {str(load_error)}")
                    continue
                finally:
                # removing STL file after its usage
                    if os.path.exists(seg_path):
                        os.remove(seg_path)

        # displaying 3D model in the plotter
            if volume:
                self.plotter.show(volume, axes=1)
                self.log_message("3D visualization has been rendered successfully.")
            else:
                self.log_message("No volumes were generated for visualization.")
                QMessageBox.warning(self, "Visualization Warning", "No volumes were generated for visualization.")

            self.plotter.background("#F5F5F5")
            self.vtk_widget.update()

        except Exception as e:
            error_message = f"Error rendering 3D visualization: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Visualization error", error_message)


    # function for loading a trained model to 3D organ visualization.
    def load_segmentation_model(self) -> torch.nn.Module:

        model_path = "best_metric_model1.pth"
        try:
            # let us initialize the model with the same parameters used during training.
            model = get_unet_model(num_classes=118, in_channels=1)
            state_dict = torch.load(model_path, map_location=torch.device('cpu')) # loading state_dict.
            model.load_state_dict(state_dict) # loading it into the model.
            model.eval() # setting model to evaluation mode.
            
            self.log_message(f"Model has been loaded successfully from {model_path}.")
            return model
        except FileNotFoundError:
            error_message = f"ERROR: {model_path} not found."
            self.log_message(error_message)
            QMessageBox.critical(self, "Model loading error", error_message)
            raise


    # function for updating image placeholders (3 of them)with the first silce from each view.
    def update_image_placeholders(self):

        if self.scan_list_sagittal:
            visualization.display_single_slice(self.scan_top_left, self.scan_list_sagittal[0])
            self.slider_sagittal.setMaximum(len(self.scan_list_sagittal) - 1)
            self.slider_sagittal.setValue(0)

        if self.scan_list_coronal:
            visualization.display_single_slice(self.scan_top_right, self.scan_list_coronal[0])
            self.slider_coronal.setMaximum(len(self.scan_list_coronal) - 1)
            self.slider_coronal.setValue(0)

        if self.scan_list_axial:
            visualization.display_single_slice(self.scan_bottom_left, self.scan_list_axial[0])
            self.slider_axial.setMaximum(len(self.scan_list_axial) - 1)
            self.slider_axial.setValue(0)


    # function for saving the segmented CT scans.
    def on_save_segmentation(self):
        self.log_message("Save segmentation action has been triggered.")
        try:
            if self.segmentation_result is not None:
                save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Segmentation Result",
                    "segmentation_result.nii.gz",
                    "NIfTI Files (*.nii *.nii.gz)"
                )
                if save_path:
                    segmentation.save_segmentation(self.segmentation_result, self.affine, save_path)
                    self.log_message(f"Segmentation saved at {save_path}.")
                    QMessageBox.information(self, "Save Successful", f"Segmentation has been saved at:\n{save_path}")
            else:
                QMessageBox.information(self, "No segmentation", "There is no segmentation data to save.")
        except Exception as e:
            error_message = f"Error saving segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Save error", error_message)

    # function for closing the segmentation and updating the visualization.
    def on_close_segmentation(self):
        self.log_message("Close segmentation action has been triggered.")
        try:
            # clearing segmentation data.
            #self.segmented_scans = {}
            self.segmentation_result = None

            # clearing CT scans.
            self.ct_scans = None
            self.affine = None

            # clearing image placeholders.
            self.scan_list_sagittal = []
            self.scan_list_coronal = []
            self.scan_list_axial = []

            self.scan_top_left.clear()
            self.scan_top_left.setText("Sagittal view")
            self.scan_top_right.clear()
            self.scan_top_right.setText("Coronal view")
            self.scan_bottom_left.clear()
            self.scan_bottom_left.setText("Axial view")

            # resetting sliders.
            self.slider_sagittal.setValue(0)
            self.slider_sagittal.setMaximum(0)
            self.slider_coronal.setValue(0)
            self.slider_coronal.setMaximum(0)
            self.slider_axial.setValue(0)
            self.slider_axial.setMaximum(0)

            # clearning 3D visualization.
            if hasattr(self, 'plotter'):
                self.plotter.clear()
                self.vtk_widget.update()

            # we must also disable segmentation action.
            self.segment_action.setEnabled(False)

            self.log_message("Segmentation data and CT scans have been cleared. Application reset to initial state.")

        except Exception as e:
            error_message = f"Error closing segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Close error", error_message)

    # function for handling 'Manage view' action from the menu bar.
    def on_manage_view(self):
        self.log_message("Manage view action has been triggered.")
        help_window = ManageViewWindow(organs=self.organs)
        help_window.signal.connect(self.on_help_window_apply)
        help_window.show()
        self.adjust_3d_visualization()

    def on_help_window_apply(self, organs):  # <-- This is the main window's slot
        self.chosen_organs = organs

    def adjust_3d_visualization(self):
        try:
            self.vtk_widget.clear() # at first, we clear the wiget to prepare for a new rendering.
            self.log_message("Clearning time. Rendering a 3D visualization...")

            self.plotter = Plotter(qt_widget=self.vtk_widget)
            self.plotter.background("#F5F5F5")

            volume = []
            #colors_rgb = self.get_distinct_colors(len(self.chosen_organs)) # generating unique colors.

            for i, organ_name in enumerate(self.chosen_organs):
                color_hex = class_to_color.get(i, "#FFFFFF")  # Domyślnie biały
                rgb_color = to_rgb(color_hex)

                seg_path = f'segmented_{organ_name}.stl'
                vol = load(seg_path).color(rgb_color)
                volume.append(vol)

                # displaying 3D visualization using afornemtioned vtk_widget library.
                self.chosen_organs = self.organs
                self.plotter = Plotter(qt_widget=self.vtk_widget)
                self.plotter.show(volume, axes=1)
                self.log_message("3D visualization has been rendered successfully.")

            self.plotter.background("#F5F5F5")
            self.vtk_widget.update() # refersh vtk_widget to see updates.
        except Exception as e:
            error_message = f"Error rendering 3D visualization: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Visualization error", error_message)

    # function for handling 'Zoom in' action from the menu bar.
    def on_zoom_in(self):
        self.log_message("Zoom in action has been triggered.")
        try:
            if hasattr(self, 'plotter'):
                self.plotter.zoom(1.2)  # zoom in by a factor 1.2.
                self.plotter.render()
                self.vtk_widget.update()
            else:
                self.log_message("No plotter available for zooming.")
        except Exception as e:
            error_message = f"Error during zoom in: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Zoom In Error", error_message)

    # function for handling 'Zoom in' action from the menu bar.
    def on_zoom_out(self):
        self.log_message("Zoom out action has been triggered.")
        try:
            if hasattr(self, 'plotter'):
                self.plotter.zoom(0.8)  # zoom out by a factor 0.8.
                self.plotter.render()
                self.vtk_widget.update()
            else:
                self.log_message("No plotter available for zooming.")
        except Exception as e:
            error_message = f"Error during zoom out: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Zoom Out Error", error_message)

    # function for displaying help message.
    def on_help(self):
        self.log_message("Help action has been triggered.")
        QMessageBox.information(self, "Help", "SegMed help information:\n\n1. Upload CT scans.\n2. Perform segmentation.\n3. Visualize results.")

    # function for displaying report problem diagnostic information.
    def on_report_problem(self):
        self.log_message("Report problem action has been triggered.")
        QMessageBox.information(self, "Report problem", "Please provide us with the details of the problem you encountered.")


    # function displaying "About" dialog with application name, its version and authors.
    def on_about(self):
        self.log_message("About action has been triggered.")

        dialog = QDialog(self)
        dialog.setWindowTitle("About")
        dialog.setFixedSize(400,300)

        about_text = """
        <div style = "text-align: center; font-size: 16px;">
        <b><i>SegMed</i></b><br><br>
        <b><i>Version 1.1</i></b><br><br>
        Authors: 
        <ul style="list-style-type: disc; padding-left: 20px; text-align: left;">
            <li>Aleksandra Kłos</li>
            <li>Olga Czajkowska</li>
            <li>Magdalena Leymańczyk</li>
        </ul>
        </div>
        """

        layout = QVBoxLayout(dialog)
        label = QLabel(about_text)
        label.setTextFormat(Qt.RichText)
        layout.addWidget(label)

        dialog.exec_()

    # function for handling image slider movement for each image placeholder.
    def on_slider_move(self):
        try:
            current_sagittal = self.slider_sagittal.value()
            current_coronal = self.slider_coronal.value()
            current_axial = self.slider_axial.value()
            
            if current_sagittal < len(self.scan_list_sagittal):
                visualization.display_single_slice(self.scan_top_left, self.scan_list_sagittal[current_sagittal])
            if current_coronal < len(self.scan_list_coronal):
                visualization.display_single_slice(self.scan_top_right, self.scan_list_coronal[current_coronal])
            if current_axial < len(self.scan_list_axial):
                visualization.display_single_slice(self.scan_bottom_left, self.scan_list_axial[current_axial])

        except Exception as e:
            error_message = f"Error moving sliders: {str(e)}"
            self.log_message(error_message)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = MedicalImageViewer()
    sys.exit(app.exec_()) 