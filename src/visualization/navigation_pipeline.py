'''
A small pipeline for navigation of already cleaned data sets.
Last update: 06/21/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import numpy as np
from src.shared_funcs.tifffunc import read_tiff
import src.visualization.stat_present as stat_present
from src.visualization.brain_navigation import region_view
import matplotlib.pyplot as plt

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

# --------------------------Below is the class of pipeline
class pipeline(object):
    '''
    0.load cleaned data
    1.load the reference image stack
    2.plot the raster cellular activities
    3.plot the anatomical distribution of the cells
    '''
    def __init__(self, rd_path ):
        '''
        load the data from the work_folder
        '''
        self._coord = None
        self._signal = None
        self._refim = None
        self._pxl = None
        try:
            raw_data = np.load(rd_path)
            self._parse_data_(raw_data)
        except IOError:
            print("The file does not exist. Please check the path! ")
            sys.exit(1) # try system exit

    def _parse_data_(self, data, verbose = True):
        '''
        Analyze the data
        '''
        try:
            self.coord = data['coord']
        except KeyError:
            print("The data does not contain correct coordinates.")
            sys.exit(1)

        try:
            self.signal = data['signal']
        except KeyError:
            print("The data does not contain dff values.")
            sys.exit(1)

        NT, NP = self.data.shape
        if verbose:
            print("# of time points:", NT, "# of cells:", NP)

    # ----------------------property members and their setters -----------------

    @property
    def coord(self):
        return self._coord
    @coord.setter
    def coord(self, new_coord):
        self._coord = new_coord

    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self, new_signal):
        self._signal = new_signal

    @property
    def refim(self):
        return self._refim
    @refim.setter
    def refim(self, new_refim):
        self._refim = new_refim

    @property
    def pxl(self):
        return self._pxl
    @pxl.setter
    def pxl(self, new_pxsize):
        self._pxl = new_pxsize

    # ----------------------simple display and save functions
    def display_select(self, ndisp, figpath = None):
        '''
        display the most active ndisp neurons
        '''
        fig = stat_present.nature_style_dffplot(self.signal[:,ndisp], dt = 0.5, sc_bar = 0.50)
        if figpath is None:
            return fig
        else:
            fig.savefig(figpath)

    def display_raster(self, ndisp = None, figpath = None):
        '''
        raster-display the neuronal activities
        '''
        if ndisp is None:
            fig = stat_present.dff_rasterplot(self.signal)
        else:
            fig = stat_present.dff_rasterplot(self.signal[:,ndisp])

        if figpath is None:
            return fig
        else:
            fig.savefig(figpath)


# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    raw_fname = global_datapath+'Jun13_A3_GCDA/'
    ZD_stack = read_tiff(raw_fname+'A3_ZD_before.tif').astype('float64')

    clean_data = np.load(raw_fname+'cleaned.npz')
    rgview = region_view(clean_data['coord'], clean_data['signal'], ZD_stack)
    rgview.show_cell(20)
    coord_active = clean_data['coord'][-20:,:]
    print(coord_active)
    rgview.fig_save(raw_fname+'first2000')


if __name__ == '__main__':
    main()
