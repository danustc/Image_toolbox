'''
A small pipeline for network analysis. Added by Dan on 06/18/2017.
Last update: 06/21/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import numpy as np
import src.dynamics.df_f as df_f # the functions of calculating dff
from src.shared_funcs.tifffunc import read_tiff
import src.visualization.stat_present as stat_present
from src.visualization.brain_navigation import region_view
import src.networks.pca_sorting as pca_sorting
import src.networks.ica_sorting as ica_sorting
import src.networks.noise_removal as noise_removal
import matplotlib.pyplot as plt

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    '''
    def __init__(self, data, raw = True, dt=0.5):
        self._coord = None
        self._signal = None
        self.dt = dt
        self.parse_data(data)
        if raw:
            self.dff_munging()

    def parse_data(self, data):
        try:
            self.coord = data['coord']
            self.signal= data['data']
        except KeyError:
            print('Wrong data!')
            self.coord = None
            self.signal = None

    #-----------------------Below are the property members-----------------

    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self, new_signal):
        self._signal= new_signal

    @property
    def coord(self):
        return self._coord
    @coord.setter
    def coord(self, new_coord):
        self._coord = new_coord

    #-----------------------Below are the data munging/analyzing functions ----------------------- 

    def get_size(self):
        return self.signal.shape

    def shuffle_data(self, ind_shuffle = None):
        '''
        shuffle the signal orders and the coordinate orders so that the position is not biasing the groupwise pca output.
        '''
        if ind_shuffle is None:
            NT, NP = self.get_size()
            ind_shuffle = np.arange(NP)
            np.random.shuffle(ind_shuffle)

        self.coord = self.coord[ind_shuffle, :]
        self.signal = self.signal[:, ind_shuffle]
        # Question: does self._data change in this way?


    def dff_munging(self, fw = 4, nt = 10, filt = True):
        '''
        calculate the df_f over the whole data set and save it
        '''
        raw_signal = df_f.dff_raw(self.signal, ft_width = fw, ntruncate = nt)[0]
        if filt:
           self.signal, self.wd = df_f.dff_expfilt(raw_signal, dt = self.dt, t_width = 1.0)
        else:
           self.signal = raw_signal


    def pca_layered_sorting(self,var_cut = 0.99, shuffle = True, verbose = True):
        '''
        perform the layered pca sorting on the data and shrink the size
        '''
        if shuffle:
            self.shuffle_data()

        NT, NP = self.get_size()
        var_ratio = var_cut/(1.-var_cut)
        gpca = pca_sorting.group_pca(self.signal, gvar = var_cut)
        gpca.group_division()
        ipre, idis = gpca.subgroup_pca(verbose)

        sig_head = self.signal[:,ipre].T
        sig_tail = self.signal[:,idis].T
        var_head = np.trace(np.cov(sig_head))
        var_tail = np.trace(np.cov(sig_tail))
        cut_ratio = var_head/var_tail

        if verbose:
            print("Initial data dimension:", self.get_size())
            print("# of discarded cells:", idis.size)
            print("Cut ratio:", cut_ratio)

        while(cut_ratio > var_ratio):
            print(ipre.size)
            self.dff_cleaning(ipre) # after dff_cleaning, self.data is updated. 
            if shuffle:
                self.shuffle_data()
            NT, NP = self.get_size()
            gpca.data = self.signal
            gpca.group_division()
            ipre, idis = gpca.subgroup_pca(fine_sort = True)
            sig_head = self.signal[:,ipre].T
            sig_tail = self.signal[:,idis].T
            var_shift = np.trace(np.cov(sig_tail))
            var_tail += var_shift
            var_head -= var_shift
            #var_head = np.trace(np.cov(sig_head))
            cut_ratio = var_head/var_tail
            if verbose:
                print("# of discarded cells:", idis.size)
                print("# current data dimension:", self.get_size())
                print("Cut ratio:", cut_ratio)

        CT, V = pca_sorting.pca_raw(self.signal, var_cut)
        a_sorted = pca_sorting.cell_sorting(V)
        self.shuffle_data(ind_shuffle = a_sorted) #rank cell by the final activities
        if verbose:
            print("Final data dimension:", self.get_size())



    def dff_cleaning(self, ipreserve):
        '''
        clean the dataset by sorting the preserved indices
        '''
        new_signal = self.signal[:,ipreserve]
        new_coord = self.coord[ipreserve, :] # select coordinates and signals
        self.signal = new_signal
        self.coord = new_coord


    def save_cleaned(self, save_path):
        '''
        Compile a dictionary and save it
        '''
        cleaned_data = dict()
        cleaned_data['coord'] = self.coord
        cleaned_data['signal'] = self.signal
        np.savez(save_path, **cleaned_data)


    # -------------------------visualization group--------------------------



# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    raw_fname = global_datapath+'Jun13_B2_control/'
    raw_data = np.load(raw_fname + 'merged.npz')
    ppl = pipeline(raw_data)
    ppl.pca_layered_sorting(var_cut = 0.95)
    ppl.save_cleaned(raw_fname+'cleaned')



if __name__ == '__main__':
    main()
