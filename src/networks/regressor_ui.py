import os
import sys
from PyQt5 import QtWidgets
import regressor_dialog_design

class Regressor_dialog(QtWidgets.QDialog):

    def __init__(self, dt, NT, parent = None):
        QtWidgets.QDialog.__init__(self, parent)
        self._ui = regressor_dialog_design.Ui_Dialog()
        self._ui.setupUi(self)


        self._ui.pushButton_exp.clicked.connect(self.exp_regressor)
        self._ui.pushButton_add.clicked.connect(self.add_steps)
        self._ui.pushButton_gen.clicked.connect(self.generate_regressor)
        self._ui.pushButton_cl.clicked.connect(self.clear_regressor)


    def exp_regressor(self):
        pass

    def add_steps(self):
        pass


    def generate_regressor(self):
        pass

    def clear_regressor(self):
        pass


