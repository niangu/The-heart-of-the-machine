# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_plotly_show.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(611, 573)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.Form_2 = QtWebEngineWidgets.QWebEngineView(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Form_2.sizePolicy().hasHeightForWidth())
        self.Form_2.setSizePolicy(sizePolicy)
        self.Form_2.setMinimumSize(QtCore.QSize(300, 250))
        self.Form_2.setStyleSheet("background-color: rgb(32, 74, 135);")
        self.Form_2.setObjectName("Form_2")
        self.gridLayout.addWidget(self.Form_2, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

from PyQt5 import QtWebEngineWidgets
