'''
A small pipeline for network analysis. Added by Dan on 06/18/2017.
Last update: 06/18/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import numpy as np
import src.dynamics.df_f as df_f # the functions of calculating dff
import src.visualization.stat_present as stat_present
import src.networks.pca_sorting as pca_sorting
import src.networks.ica_sorting as ica_sorting

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    '''
    def __init__(self, data, raw = True):
        self.data(data)
        if raw:
            self.dff_munging()

    def _parse_data_(self):
        try:
            self.coord = self.data['coords']
            self.fluor = self.data['data']
        except KeyError:
            print('Wrong data!')
            self.data = None

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._parse_data_()

    def dff_munging(self, fw = 4, nt = 10):
        '''
        calculate the df_f over the whole data set and save it
        '''
        dff_raw = df_f.dff_raw(self.fluor, ft_width = fw, ntruncate = nt)

    def analyze(self,var_cut = 0.95):
        pass



# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    raw_fname = global_datapath+'Jun13_A1_GCDA/'
    raw_data = np.load(raw_fname + 'merged.npz')
    ppl = pipeline(raw_data)
    ppl.data=None
