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
from model import get_unet_model  

# wigets and libraries used: https://doc.qt.io/qt-6/qtwidgets-module.html   https://doc.qt.io/qt-6/widget-classes.html


# as in case of the Dataloader, the mapping can be found here: https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file

main_classes_CT = {
    "skeleton": [
        'skull', 'clavicula_left', 'clavicula_right', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'sternum',
        'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8',
        'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4',
        'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12',
        'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7', 'vertebrae_L1',
        'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5', 'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3',
        'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8', 'vertebrae_T9', 'vertebrae_T10',
        'vertebrae_T11', 'costal_cartilages', 'vertebrae_T12', 'hip_left', 'hip_right', 'sacrum', 'femur_left', 'femur_right'
    ],
    "cardiovascular": [
        'common_carotid_artery_left', 'common_carotid_artery_right', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right',
        'subclavian_artery_left', 'subclavian_artery_right', 'brachiocephalic_trunk', 'superior_vena_cava', 'pulmonary_vein',
        'atrial_appendage_left', 'aorta', 'heart', 'portal_vein_and_splenic_vein', 'inferior_vena_cava', 'iliac_artery_left',
        'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right'
    ],
    "gastrointestinal": [
        'esophagus', 'stomach', 'duodenum', 'small_bowel', 'colon', 'urinary_bladder'
    ],
    "muscles": [
        'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right', 'gluteus_minimus_left',
        'gluteus_minimus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_maximus_left',
        'gluteus_maximus_right'
    ],
    "others": [
        'brain', 'spinal_cord', 'thyroid_gland', 'trachea', 'lung_upper_lobe_left', 'lung_upper_lobe_right',
        'lung_middle_lobe_right', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'adrenal_gland_left',
        'adrenal_gland_right', 'spleen', 'liver', 'gallbladder', 'kidney_left', 'kidney_right', 'kidney_cyst_left',
        'kidney_cyst_right', 'pancreas', 'prostate'
    ]
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


class MedicalImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ct_scans = None
        self.affine = None
        self.segmented_scans = {}
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
                colors_rgb = self.get_distinct_colors(len(unique_lbls)) # generating unique colors. 


                for i, lbl in enumerate(unique_lbls):
                    organ_name = lbl_to_organ.get(int(lbl), f'label_{int(lbl)}')
                    organ_mask = (seg_data == lbl).astype(np.uint8)

                    if np.sum(organ_mask) == 0: # skipping empty mask.
                        self.log_message(f'Skipping label {lbl}.')
                        continue

                    # geenrating STL file and load it as a 3D model.
                    seg_path = f'segmented_{organ_name}.stl'
                    demo.convert_to_stl(organ_mask, seg_path)

                    vol = load(seg_path).color(colors_rgb[i % len(colors_rgb)])
                    volume.append(vol)
                    i += 1

                    # clearning temporary STL files.
                    if os.path.exists(seg_path):
                        os.remove(seg_path)

                # displaying 3D visualization using afornemtioned vtk_widget library.
                self.plotter = Plotter(qt_widget=self.vtk_widget)
                self.plotter.show(volume, axes=1)
                self.log_message("3D visualization has been rendered successfully.")
            else:
                self.log_message("No segmentation file provided, visualization skipped.") # TODO: naprawić

            # plotter.background("white")
            self.plotter.background("#F5F5F5")
            self.vtk_widget.update() # refersh vtk_widget to see updates.
        except Exception as e:
            error_message = f"Error rendering 3D visualization: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Visualization error", error_message)


    # this function's purpose is to generate N distinct RGB colors.
    def get_distinct_colors(self, N):
        colormaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 
        'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']  # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        colors = []
    
        for i in range(N):
            cmap_name = colormaps[i % len(colormaps)]  # cycling through available colormaps defined above.
            cmap = cm.get_cmap(cmap_name)
            normalized_idx = (i % cmap.N) / cmap.N # normalize index.
            rgb_color = cmap(normalized_idx)[:3]  # extract RGB values.
            colors.append(rgb_color)
    
        return colors

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
                background-color: #FFFFFF;  /* Jasne tło dla VTK widget */
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

        # Create a grid layout for the container
        vtk_container_layout = QGridLayout()
        vtk_container_layout.setContentsMargins(0, 0, 0, 0)
        vtk_container_layout.setSpacing(0)
        self.vtk_container.setLayout(vtk_container_layout)

        # Add the vtk_widget to the grid layout
        vtk_container_layout.addWidget(self.vtk_widget, 0, 0)

        # Create the zoom in and zoom out buttons
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedSize(30, 30)
        self.zoom_in_button.clicked.connect(self.on_zoom_in)
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedSize(30, 30)
        self.zoom_out_button.clicked.connect(self.on_zoom_out)

        # Style the buttons
        button_style = """
            QPushButton {
                background-color: rgba(135, 206, 250, 180);  /* Semi-transparent */
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

        # Create a layout to hold the buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(5, 5, 5, 5)
        buttons_layout.setSpacing(5)
        buttons_layout.addWidget(self.zoom_in_button)
        buttons_layout.addWidget(self.zoom_out_button)
        buttons_layout.addStretch()
        buttons_layout.setAlignment(Qt.AlignTop | Qt.AlignRight)

        # Create a widget to hold the buttons layout
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setStyleSheet("background-color: transparent;")

        # Add the buttons_widget to the grid layout, overlaid on the vtk_widget
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
            dir_dialog = QFileDialog(self)
            dir_dialog.setWindowTitle("Select directory with segmented CT scan:")
            dir_dialog.setFileMode(QFileDialog.Directory)
            if dir_dialog.exec_():
                chosen_dir = dir_dialog.selectedFiles()
                if chosen_dir:
                    dir_path = chosen_dir[0]
                    self.log_message(f"Uploading segmented CT scan from directory {dir_path}...")

                    segmented_scans = [
                        os.path.join(dir_path, f) for f in os.listdir(dir_path)
                        if f.endswith('.nii') or f.endswith('.nii.gz')
                    ]   

                    if not segmented_scans:
                        raise ValueError("No segmented CT scans found in the chosen directory.")

                    self.segmented_scans = {}
                    for file_path in segmented_scans:
                        organ_name = os.path.splitext(os.path.basename(file_path))[0]
                        seg_data, affine = data_import.load_nifti(file_path)
                        self.segmented_scans[organ_name] = seg_data
                        self.log_message(f"Segmented CT scan for {organ_name} has been successfully uploaded.")

                    # disabling segmentation action, as "Segment image" option has been already executed.
                    self.segment_action.setEnabled(False)
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

            self.log_message("Saving the segmentation result as a NIfTI file...")
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Segmentation Result", "", "NIfTI Files (*.nii *.nii.gz)")
            if save_path:
                segmentation.save_segmentation(seg_out, self.affine, save_path)
                self.log_message(f"Segmentation has been saved at {save_path}.")

                # optionally, reloading the segmentation for visualization.
                self.render_3d_visualization(save_path)
        except Exception as e:
            error_message = f"ERROR: error while performing segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Segmentation error", error_message)


    # function for loading a trained model to 3D organ visualization.
    def load_segmentation_model(self) -> torch.nn.Module:

        model_path = "best_metric_model.pth"
        try:
            # let us initialize the model with the same parameters used during training.
            model = get_unet_model(num_classes=117, in_channels=1)
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
            if self.segmented_scans:
                for organ, seg_data in self.segmented_scans.items():
                    save_path, _ = QFileDialog.getSaveFileName(
                        self, 
                        f"Save {organ} Segmentation", 
                        f"{organ}_segmentation.nii.gz", 
                        "NIfTI Files (*.nii *.nii.gz)"
                    )
                    if save_path:
                        segmentation.save_segmentation(seg_data, self.affine, save_path)
                        self.log_message(f"Segmentation for {organ} saved at {save_path}.")
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
            self.segmented_scans = {}
            self.log_message("Segmentation data has been cleared.")
            self.render_3d_visualization()  # clear visualization.
        except Exception as e:
            error_message = f"Error closing segmentation: {str(e)}."
            self.log_message(error_message)
            QMessageBox.critical(self, "Close error", error_message)


     # function for handling 'Manage view' action from the menu bar.
    def on_manage_view(self):
        self.log_message("Manage view action has been triggered.")
        # TODO: implement

    # function for handling 'Zoom in' action from the menu bar.
    def on_zoom_in(self):
        self.log_message("Zoom in action has been triggered.")
        try:
            if hasattr(self, 'plotter'):
                self.plotter.zoom(1.2)  # Zoom in by a factor of 1.2
                self.plotter.render()
                self.vtk_widget.update()
            else:
                self.log_message("No plotter available for zooming.")
        except Exception as e:
            error_message = f"Error during zoom in: {str(e)}"
            self.log_message(error_message)
            QMessageBox.critical(self, "Zoom In Error", error_message)

    # function for handling 'Zoom in' action from the menu bar
    def on_zoom_out(self):
        self.log_message("Zoom out action has been triggered.")
        try:
            if hasattr(self, 'plotter'):
                self.plotter.zoom(0.8)  # Zoom out by a factor of 0.8
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
