# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'regressor_dialog_design.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(585, 442)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 360, 541, 53))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_ts = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_ts.setObjectName("label_ts")
        self.gridLayout.addWidget(self.label_ts, 0, 1, 1, 1)
        self.label_mode = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_mode.sizePolicy().hasHeightForWidth())
        self.label_mode.setSizePolicy(sizePolicy)
        self.label_mode.setObjectName("label_mode")
        self.gridLayout.addWidget(self.label_mode, 0, 2, 1, 1)
        self.lineEdit_pw = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_pw.setObjectName("lineEdit_pw")
        self.gridLayout.addWidget(self.lineEdit_pw, 1, 0, 1, 1)
        self.spinBox_mode = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_mode.sizePolicy().hasHeightForWidth())
        self.spinBox_mode.setSizePolicy(sizePolicy)
        self.spinBox_mode.setObjectName("spinBox_mode")
        self.gridLayout.addWidget(self.spinBox_mode, 1, 2, 1, 1)
        self.lineEdit_ts = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_ts.setObjectName("lineEdit_ts")
        self.gridLayout.addWidget(self.lineEdit_ts, 1, 1, 1, 1)
        self.label_pw = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_pw.setObjectName("label_pw")
        self.gridLayout.addWidget(self.label_pw, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.pushButton_add = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_add.sizePolicy().hasHeightForWidth())
        self.pushButton_add.setSizePolicy(sizePolicy)
        self.pushButton_add.setObjectName("pushButton_add")
        self.horizontalLayout.addWidget(self.pushButton_add)
        self.pushButton_gen = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_gen.sizePolicy().hasHeightForWidth())
        self.pushButton_gen.setSizePolicy(sizePolicy)
        self.pushButton_gen.setObjectName("pushButton_gen")
        self.horizontalLayout.addWidget(self.pushButton_gen)
        self.pushButton_cl = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_cl.sizePolicy().hasHeightForWidth())
        self.pushButton_cl.setSizePolicy(sizePolicy)
        self.pushButton_cl.setObjectName("pushButton_cl")
        self.horizontalLayout.addWidget(self.pushButton_cl)
        self.mpl_regressor = MatplotlibWidget(Dialog)
        self.mpl_regressor.setGeometry(QtCore.QRect(18, 50, 541, 301))
        self.mpl_regressor.setObjectName("mpl_regressor")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_ts.setText(_translate("Dialog", "Time stamp"))
        self.label_mode.setText(_translate("Dialog", "Mode"))
        self.label_pw.setText(_translate("Dialog", "Pulse width"))
        self.pushButton_add.setText(_translate("Dialog", "Add"))
        self.pushButton_gen.setText(_translate("Dialog", "Generate"))
        self.pushButton_cl.setText(_translate("Dialog", "Clear"))

from matplotlibwidget import MatplotlibWidget
