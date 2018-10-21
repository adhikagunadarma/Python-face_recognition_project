from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap, qRgb

import sys
import traceback

class Main(QDialog) :
    def __init__(self):
        try:
            super(Main, self).__init__()
            loadUi('Main.ui', self)
            self.setWindowTitle('Select Dataset')
            self.goButton.clicked.connect(self.on_goButton_clicked)
            self.datasetBox.addItem("Caltech")
            self.datasetBox.addItem("ORL")
        except Exception:
            print(traceback.format_exc())
            pass


    @pyqtSlot()
    def on_goButton_clicked(self):
        print("go fuckyourself")



def main():
    app = QApplication(sys.argv)
    widget = Main()
    widget.show()
    sys.exit(app.exec())

if __name__ == 'main':
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        pass

