from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication


class OrganSelectionDialog(QDialog):
    def __init__(self, organs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select an Organ")
        self.setFixedSize(300, 150)  # window's fixed size

        # window in the middle
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # layout
        layout = QVBoxLayout()

        # label
        self.label = QLabel("Choose an organ:")
        layout.addWidget(self.label)

        # dropdown (ComboBox)
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(organs)  # full organ names
        layout.addWidget(self.combo_box)

        # buttons (OK, Cancel)
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    # function to get (return) the selected organ
    def get_selected_organ(self):
        return self.combo_box.currentText()
