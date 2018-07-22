import sys
import os
from PyQt5 import QtGui, QtWidgets, QtCore
import network_pipeline
import numpy as np
from functools import partial
from collections import deque
from network_pipeline import pipeline as nw_pipeline
from dff_pipeline import pipeline as dff_pipeline
from src.visualization import stat_present, signal_plot
import network_design
from regressor_ui import Regressor_dialog

global_datapath = '/home/sillycat/Programming/Python/data_test/'

class UI(object):
    def __init__(self):
        '''
        initialize the UI.
        '''
        self.network_pipeline = nw_pipeline()
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QMainWindow()
        self._window.closeEvent = self.shutDown

        self._ui = network_design.Ui_Form()
        self._ui.setupUi(self._window)
        self.data_loaded = False
        self.work_folder = None
        self.fig_empty = True


        # setup the connection between the buttons and texts
        self._ui.pushButton_dff.clicked.connect(self.display_signal)
        self._ui.pushButton_dff.clicked.connect(self.noise_level)
        self._ui.pushButton_ica.clicked.connect(self.ica)
        self._ui.pushButton_pcas.clicked.connect(self.pcas)
        self._ui.pushButton_kmeans.clicked.connect(self.kmeans)
        self._ui.pushButton_regress.clicked.connect(self.regression)
        self._ui.pushButton_loaddata.clicked.connect(self.load_data)
        self._ui.pushButton_expsig.clicked.connect(self.figure_export)
        self._ui.pushButton_regdes.clicked.connect(self.set_regressor)
        self._ui.pushButton_deln.clicked.connect(self.del_neurons)
        self._ui.lineEdit_numneurons.returnPressed.connect(self.set_numNeurons)


        # Initialize by running a couple of functios
        self.set_numNeurons()

        self._window.show()
        self._app.exec_()

    def load_data(self):
        '''
        load data with self.network_pipeline.parse_data.
        Data must be read from the network_pipeline.
        '''
        fname, _ =QtWidgets.QFileDialog.getOpenFileName(None, directory = global_datapath, caption = 'data file to load:')
        self._ui.lineEdit_data.setText(fname)
        self.network_pipeline.parse_data(data_file = fname)
        self.work_folder = os.path.dirname(fname)
        self.basename = os.path.basename(fname).split('.')[0]
        self.data_loaded = True
        self.dt = float(self._ui.lineEdit_tstep.text()) # time step
        self.NT = self.network_pipeline.get_size()[0]


    def ica(self):
        if self.data_loaded:
            self.n_ica = self._ui.spinBox_nICs.value()
            self.n_clu = self._ui.spinBox_nclus.value()
            self.network_pipeline.ica_clustering(c_fraction = 0.50,n_components = self.n_ica, n_clusters = self.n_clu)
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
            self.network_pipeline.pca_layered_sorting(var_cut = 0.95, verbose = True)



    def kmeans(self):
        pass

    def regression(self):
        '''
        regress the loaded data to the determined regressor.
        '''

    def set_numNeurons(self):
        disp_range = list(map(int, self._ui.lineEdit_numneurons.text().split(',')))
        print(disp_range)
        if len(disp_range)==1:
            self.nu_start = 0
            self.nu_end = int(disp_range[0])
            self.nu_step = 1
        else:
            self.nu_start = int(disp_range[0])
            self.nu_end = int(disp_range[1])
            self.nu_step = 1

        if len(disp_range) ==3:
            self.nu_step = int(disp_range[2]) # step_size


    def set_regressor(self):
        '''
        custom design of regressor.
        '''
        regressor_dialog = Regressor_dialog(self.dt, self.NT)
        if regressor_dialog.exec_():
            print("Regressor design:")

    # ------------------------------Below are a couple of visualization functions -------------------------------
    def display_signal(self):
        '''
        display signals
        '''
        sind = np.arange(self.nu_start, self.nu_end, self.nu_step)
        signal_show = self.network_pipeline.get_cells_index(sind)[0]

        self._ui.mpl_signal.figure.clf()
        signal_plot.dff_rasterplot(signal_show, fig = self._ui.mpl_signal.figure)
        self._ui.mpl_signal.draw()

    def display_ICs(self):
        '''
        OK This works!
        '''
        icas = self.network_pipeline.ic
        NT, NC = icas.shape
        print(NT, NC)
        for ii in range(NC):
            ax = self._ui.mpl_analysis.figure.add_subplot(NC, 1, ii+1)
            ax.plot(icas[:,ii], '-g')
            ax.set_xticklabels([])
        self._ui.mpl_analysis.draw()
        self.fig_empty = False


    def display_hist(self):

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


    def shutDown(self, event):
        self.data = None
        self._app.quit()



def main():
    nw_ui = UI()

if __name__ == '__main__':
    main()
