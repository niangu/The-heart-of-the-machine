from ui_savefile import Ui_Dialog
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog, QApplication
from PyQt5.QtCore import pyqtSlot, QCoreApplication

import sys
import pandas

class SaveFile(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(SaveFile, self).__init__(parent)
        #self.ui_savefile = Ui_Dialog()
        self.setupUi(self)
        """
        self.sep = ','
        self.na_rep = ''
        self.encoding = 'utf-8'
        self.compression = 'infer'
        self.decimal = '.'
        self.savefilename = ""
        self.index = True
        self.header = True
        """
        #self.Cancel_Btn.clicked.connect(self.Saveclicked)
    @pyqtSlot()
    def on_file_Btn_clicked(self):
        filename, type = QFileDialog.getSaveFileName(self, "保存文件", "data.csv", "Text File(*.csv)")
        self.lineEdit_file.setText(filename)

    @pyqtSlot()
    def on_save_Btn_clicked(self):
        self.sep = self.lineEdit_sep.text()
        self.na_rep = self.lineEdit_na_rep.text()
        self.encoding = self.lineEdit_encoding.text()
        self.compression = self.comboBox_compression.currentText()
        self.decimal = self.lineEdit_decimal.text()
        self.savefilename = self.lineEdit_file.text()
        self.index = self.checkBox_index.isChecked()
        self.header = self.checkBox_header.isChecked()
        if len(self.savefilename) == 0:
            QMessageBox.critical(self, '提示', "请选择保存路径", QMessageBox.Ok)
            return
        self.accept()
    @pyqtSlot()
    def on_Cancel_Btn_clicked(self):
        self.close()

        
    def save(self):
        sep = self.sep
        na_rep = self.na_rep
        encoding = self.encoding
        savefilename = self.savefilename
        if self.compression == 'None':
            compression = None
        else:
            compression = self.compression
        if self.compression == 'gzip':
            savefilename = self.savefilename + '.gzip'
        decimal = self.decimal

        index = self.index
        header = self.header
        return sep, na_rep, encoding, compression, decimal, savefilename, index, header



