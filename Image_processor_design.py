# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Image_processor_design.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(791, 705)
        Form.setMouseTracking(True)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(40, 50, 161, 471))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_pxl = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_pxl.setObjectName("label_pxl")
        self.verticalLayout.addWidget(self.label_pxl)
        self.lineEdit_pxl = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_pxl.setObjectName("lineEdit_pxl")
        self.verticalLayout.addWidget(self.lineEdit_pxl)
        self.label_roidia = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_roidia.setObjectName("label_roidia")
        self.verticalLayout.addWidget(self.label_roidia)
        self.lineEdit_roidia = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_roidia.setObjectName("lineEdit_roidia")
        self.verticalLayout.addWidget(self.lineEdit_roidia)
        self.label_thresh = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_thresh.setObjectName("label_thresh")
        self.verticalLayout.addWidget(self.label_thresh)
        self.lineEdit_thresh = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_thresh.setObjectName("lineEdit_thresh")
        self.verticalLayout.addWidget(self.lineEdit_thresh)
        self.label_segmet = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_segmet.setObjectName("label_segmet")
        self.verticalLayout.addWidget(self.label_segmet)
        self.listWidget_segmet = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.listWidget_segmet.setObjectName("listWidget_segmet")
        self.verticalLayout.addWidget(self.listWidget_segmet)
        self.label_basewin = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_basewin.setObjectName("label_basewin")
        self.verticalLayout.addWidget(self.label_basewin)
        self.lineEdit_basewin = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_basewin.setObjectName("lineEdit_basewin")
        self.verticalLayout.addWidget(self.lineEdit_basewin)
        self.label_dt = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_dt.setObjectName("label_dt")
        self.verticalLayout.addWidget(self.label_dt)
        self.lineEdit_dt = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_dt.setObjectName("lineEdit_dt")
        self.verticalLayout.addWidget(self.lineEdit_dt)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(220, 50, 531, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_file = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_file.setObjectName("pushButton_file")
        self.horizontalLayout.addWidget(self.pushButton_file)
        self.lineEdit_file = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit_file.setObjectName("lineEdit_file")
        self.horizontalLayout.addWidget(self.lineEdit_file)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(220, 100, 531, 491))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.mpl_image = MatplotlibWidget(self.verticalLayoutWidget_2)
        self.mpl_image.setObjectName("mpl_image")
        self.verticalLayout_2.addWidget(self.mpl_image)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_expfig = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_expfig.setObjectName("pushButton_expfig")
        self.horizontalLayout_2.addWidget(self.pushButton_expfig)
        self.lineEdit_expfig = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_expfig.setObjectName("lineEdit_expfig")
        self.horizontalLayout_2.addWidget(self.lineEdit_expfig)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(220, 610, 531, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_seg = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_seg.sizePolicy().hasHeightForWidth())
        self.pushButton_seg.setSizePolicy(sizePolicy)
        self.pushButton_seg.setObjectName("pushButton_seg")
        self.horizontalLayout_3.addWidget(self.pushButton_seg)
        self.pushButton_dff = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_dff.sizePolicy().hasHeightForWidth())
        self.pushButton_dff.setSizePolicy(sizePolicy)
        self.pushButton_dff.setObjectName("pushButton_dff")
        self.horizontalLayout_3.addWidget(self.pushButton_dff)
        self.pushButton_sort = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_sort.sizePolicy().hasHeightForWidth())
        self.pushButton_sort.setSizePolicy(sizePolicy)
        self.pushButton_sort.setObjectName("pushButton_sort")
        self.horizontalLayout_3.addWidget(self.pushButton_sort)
        self.pushButton_zs = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_zs.sizePolicy().hasHeightForWidth())
        self.pushButton_zs.setSizePolicy(sizePolicy)
        self.pushButton_zs.setObjectName("pushButton_zs")
        self.horizontalLayout_3.addWidget(self.pushButton_zs)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_pxl.setText(_translate("Form", "Pixel size (micron)"))
        self.label_roidia.setText(_translate("Form", "ROI diameter (pixels)"))
        self.label_thresh.setText(_translate("Form", "Threshold"))
        self.label_segmet.setText(_translate("Form", "Segment method"))
        self.label_basewin.setText(_translate("Form", "Baseline window (s)"))
        self.label_dt.setText(_translate("Form", "Time step (s)"))
        self.pushButton_file.setText(_translate("Form", "Load image"))
        self.pushButton_expfig.setText(_translate("Form", "Save figure"))
        self.pushButton_seg.setText(_translate("Form", "Segmentation"))
        self.pushButton_dff.setText(_translate("Form", "DF/F"))
        self.pushButton_sort.setText(_translate("Form", "Sort"))
        self.pushButton_zs.setText(_translate("Form", "Z-score"))

from matplotlibwidget import MatplotlibWidget
