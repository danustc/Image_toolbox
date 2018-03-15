import sys
import os
from PyQt5 import QtGui, QtWidgets, QtCore
import network_pipeline
import numpy as np
from functools import partial
from collections import deque


class network_ui(object):
    def __init__(self):
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QWidget()
        self._window.closeEvent = self.shutDown

        self._ui = network_design.Ui_Form()
        self._ui.setupUi(self._window)

    def shutDown(self):
        pass
