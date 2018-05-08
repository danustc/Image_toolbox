import sys
import os
from PyQt5 import QtGui, QtWidgets, QtCore
import network_pipeline
import numpy as np
from functools import partial
from collections import deque
from network_pipeline import pipeline
from src.visualization import stat_present, signal_plot
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
        self.data_loaded = False
        self.work_folder = None
        self.fig_empty = True


        # setup the connection between the buttons and texts
        self._ui.pushButton_ica.clicked.connect(self.ica)
        self._ui.pushButton_pcas.clicked.connect(self.pcas)
        self._ui.pushButton_kmeans.clicked.connect(self.kmeans)
        self._ui.pushButton_regress.clicked.connect(self.regression)
        self._ui.pushButton_loaddata.clicked.connect(self.load_data)
        self._ui.pushButton_expfig.clicked.connect(self.figure_export)
        self._ui.pushButton_display.clicked.connect(self.display_signal)
        self._ui.lineEdit_numneurons.returnPressed.connect(self.set_numNeurons)

        self._window.show()
        self._app.exec_()

    def load_data(self):
        fname, _ =QtWidgets.QFileDialog.getOpenFileName(None,'data file to load:')
        self._ui.lineEdit_data.setText(fname)
        self._pipeline.parse_data(data_file = fname)
        self.work_folder = os.path.dirname(fname)
        self.basename = os.path.basename(fname).split('.')[0]
        self.data_loaded = True


    def ica(self):
        if self.data_loaded:
            self.n_ica = self._ui.spinBox_nICs.value()
            self.n_clu = self._ui.spinBox_nclus.value()
            self._pipeline.ica_clustering(c_fraction = 0.50,n_components = self.n_ica, n_clusters = self.n_clu)
            print("Finished!")
            self.display_ICs()
        else:
            print("Please load data first.")

    def pcas(self):
        '''
        perform layered PCA-sorting on the DF/D data.
        '''
        if self.data_loaded:
            vcut = float(self._ui.lineEdit_vcut.text())
            self._pipeline.pca_layered_sorting(var_cut = 0.95, verbose = True)


    def kmeans(self):
        pass

    def regression(self):
        '''
        regress the loaded data to the determined regressor.
        '''

    def set_numNeurons(self):
        pass

    # ------------------------------Below are a couple of visualization functions -------------------------------
    def display_signal(self):
        '''
        display signals
        '''
        disp_range = list(map(int, self._ui.lineEdit_numneurons.text().split(',')))
        print(disp_range)
        if len(disp_range)==1:
            sind = np.arange(disp_range[0])
        else:
            sind = np.arange(int(disp_range[0]), int(disp_range[1]))
        signal_show = self._pipeline.get_cells_index(sind)[0]

        self._ui.mpl_analysis.figure.clf()
        signal_plot.dff_rasterplot(signal_show, fig = self._ui.mpl_analysis.figure)
        self._ui.mpl_analysis.draw()

    def display_ICs(self):
        '''
        OK This works!
        '''
        icas = self._pipeline.ic
        NT, NC = icas.shape
        print(NT, NC)
        for ii in range(NC):
            ax = self._ui.mpl_analysis.figure.add_subplot(NC, 1, ii+1)
            ax.plot(icas[:,ii], '-g')
            ax.set_xticklabels([])
        self._ui.mpl_analysis.draw()
        self.fig_empty = False

    def figure_export(self):
        '''
        export current figure
        '''
        if self.fig_empty:
            print("No figure to be saved.")
            return
        else:
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(None,'Save as:', self.work_folder+'/'+self.basename + '*.png', "Images (*.png, *.jpg)"  )
            self._ui.mpl_analysis.figure.savefig(fname)

    def display_dff_raster(self):
        pass

    def shutDown(self, event):
        self.data = None
        self._app.quit()



def main():
    nw_pipeline = pipeline()
    nw_ui = UI(nw_pipeline)

if __name__ == '__main__':
    main()
