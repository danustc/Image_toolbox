'''
This is the ui interface for data processing pipeline.
Created by Dan on 04/28/2018.
'''

from PyQt5 import QtWidgets, QtCore
import sys
import os
import numpy as np
from Image_processor import pipeline
import Image_processor_design
import tifffile as tf

class UI(object):
    def __init__(self):
        '''
        initialize the UI.
        '''
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = QtWidgets.QMainWindow()
        self._window.closeEvent = self.shutDown

        self._ui = Image_processor_design.Ui_Form()
        self._ui.setupUi(self._window)
        self.data_loaded = False
        self.work_folder = None
        self.image = None
        self.fig_empty = True


        # setup the connection between the buttons and texts
        self._window.show()
        self._app.exec_()


    def _load_data_(self, fpath):
        image = tf.imread(fpath)
        nz, ny, nx = image.shape
        if(nz*ny*nx>0):
            self.image = image
            self.data_loaded = True


    def segmentation(self, mode = 'i'):
        '''
        segmentation of images from the sample picture
        '''
        pass


    def display_image(self):
        pass

    def shutDown(self, event):
        self._app.quit()
def main():
    nw_ui = UI()

if __name__ == '__main__':
    main()
