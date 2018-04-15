import sys
import os
from PyQt5 import QtGui, QtWidgets, QtCore
import network_pipeline
import numpy as np
from functools import partial
from collections import deque
from network_pipeline import pipeline
import network_design


class UI(object):
    def __init__(self, pl):
        '''
        initialize the UI.
        '''
        self._pipeline = pl
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QMainWindow()
        self._window.closeEvent = self.shutDown

        self._ui = network_design.Ui_Form()
        self._ui.setupUi(self._window)


        # setup the connection between the buttons and texts
        self._ui.pushButton_ica.clicked.connect(self.ica)
        self._ui.pushButton_pcas.clicked.connect(self.pcas)
        self._ui.pushButton_kmeans.clicked.connect(self.kmeans)
        self._ui.pushButton_trigger.clicked.connect(self.trigger_filt)
        self._ui.pushButton_load.clicked.connect(self.load_data)
        self._ui.lineEdit_numneurons.returnPressed.connect(self.set_numNeurons)

        self._window.show()
        self._app.exec_()

    def load_data(self):
        fname, _ =QtWidgets.QFileDialog.getOpenFileName(None,'data file to load:')
        self._pipeline.parse_data(data_file = fname)


    def ica(self):
        self.n_ica = self._ui.spinBox_nICs.value()
        self.n_clu = self._ui.spinBox_nclus.value()
        self._pipeline.ica_clustering(c_fraction = 0.50,n_components = self.n_ica, n_clusters = self.n_clu)
        print("Finished!")

    def pcas(self):
        pass

    def kmeans(self):
        pass

    def trigger_filt(self):
        pass

    def set_numNeurons(self):
        pass

    def shutDown(self, event):
        self._app.quit()



def main():
    nw_pipeline = pipeline()
    nw_ui = UI(nw_pipeline)

if __name__ == '__main__':
    main()
