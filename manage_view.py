# === manage_view.py ===

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout

class OrganSelectionDialog(QDialog):
    """ 
    Class for selecting organ from the manage_view menu.
    """
    def __init__(self, organs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select an organ")
        self.setFixedSize(300, 150)  # window's fixed size

        # window in the middle:
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # layout:
        layout = QVBoxLayout()

        # label:
        self.label = QLabel("Choose an organ:")
        layout.addWidget(self.label)

        # dropdown (ComboBox):
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(organs)  # full organ names
        layout.addWidget(self.combo_box)

        # buttons (OK, Cancel):
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.was_closed = True  # if dialog window was closed

    
    def closeEvent(self, event):
        """ 
        Method for closing the dialog window.
        """

        self.was_closed = True # dialog window closed
        super().closeEvent(event)


    def accept(self):
        """ Method for accepting the selected organ.
        """

        self.was_closed = False # dialog window accepted
        super().accept()


    def get_selected_organ(self):
        """A getter method to return (get) the selected organ from the menu.
         """

        return self.combo_box.currentText()
