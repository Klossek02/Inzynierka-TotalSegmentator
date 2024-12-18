from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton


class ManageViewWindow(QWidget):
    def __init__(self, parent=None, organs=None):
        super().__init__(parent)
        if organs is None or len(organs) == 0:
            organs = []
        self.organs = organs
        self.checkbox_list = {}
        layout = QVBoxLayout()
        self.label = QLabel("Another Window")
        layout.addWidget(self.label)
        self.init_checkbox_list()
        self.setLayout(layout)

        self.applyButton = QPushButton("Apply")
        self.layout().addWidget(self.applyButton)

        self.signal = pyqtSignal(list)

    def init_checkbox_list(self):
        for org in self.organs:
            self.checkbox_list.update({org: QCheckBox(text=org)})
            self.layout().addWidget(self.checkbox_list[org])

    def on_press_applyButton(self):
        self.signal.emit(self.organs)
        self.close()
