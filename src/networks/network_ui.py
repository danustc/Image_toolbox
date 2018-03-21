import sys
import os
from PyQt5 import QtGui, QtWidgets, QtCore
import network_pipeline
import numpy as np
from functools import partial
from collections import deque
import network_design


class network_ui(object):
    def __init__(self, control):
        '''
        initialize the UI.
        '''
        self._control = control
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QWidget()
        self._window.closeEvent = self.shutDown

        self._ui = network_design.Ui_Form()
        self._ui.setupUi(self._window)


        # setup the connection between the buttons and texts
        self._ui.pushButton_ica.clicked.connect(self.ica)
        self._ui.pushButton_pcas.clicked.connect(self.pcas)
        self._ui.pushButton_kmeans.clicked.connect(self.kmeans)
        self._ui.pushButton_trigger.clicked.connect(self.trigger_filt)

        self._ui.lineEdit_numneurons.returnPressed.connect(self.num_neurons)



    def ica(self):
        pass

    def pcas(self):
        pass

    def kmeans(self):
        pass

    def trigger_filt(self):
        pass

    def shutDown(self):
        pass
