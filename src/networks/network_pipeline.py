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
import src.networks.simple_variance as simple_variance
import src.networks.noise_removal as noise_removal
#from src.networks.temporal_sorting import time_window_section
import matplotlib.pyplot as plt

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    1. Do the sorting/cleaning if necessary.
    2. Do the detailed analysis if necessary.
    Conventions:    a. self.signal stores signals (either F or \Delta F/F) of all the extracted cells.
                    b. self.cood stores the coordinates (in micron) of all the extracted cells. If coord has two columns, the 0th is y and the 1st is x; if coord has three columns, the order is z,y,x (instead of x,y,z, to be consistent with Python array dimension orders)
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
            try:
                self.signal = data['data']
            except KeyError:
                try:
                    self.signal = data['signal']
                except KeyError:
                    print("No signal data stored in the file!")
                    sys.exit(1)
        except KeyError:
            print('Wrong data!')
            self.coord = None
            self.signal = None
            sys.exit(1)

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

    def get_cells_index(self, nc_groups):
        return self.signal[:,nc_groups], self.coord[nc_groups, :]

    def get_cells_space(self, pos_center, radius):
        '''
        get a list of cells within the desired region.
        '''
        pass # to be filled later


    def svar_sorting(self, var_cut = 0.95):
        '''
        simple variance-based sorting
        '''
        simple_variance.simvar_global_sort(self.signal)


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

        sig_head = self.signal[:,ipre]
        sig_tail = self.signal[:,idis]
        var_head = np.sum(np.var(sig_head, axis = 0))
        var_tail = np.sum(np.var(sig_tail, axis = 0))
        cut_ratio = var_head/var_tail

        if verbose:
            print("Initial data dimension:", self.get_size())
            print("# of discarded cells:", idis.size)
            print("Cut ratio:", cut_ratio)

        while(cut_ratio > var_ratio):
            print(ipre.size)
            self.dff_cleaning(ipre) # after dff_cleaning, self.signal is updated. 
            if shuffle:
                self.shuffle_data()
            NT, NP = self.get_size()
            gpca.data = self.signal
            gpca.group_division()
            ipre, idis = gpca.subgroup_pca(fine_sort = True)
            sig_head = self.signal[:,ipre]
            sig_tail = self.signal[:,idis]
            var_shift = np.sum(np.var(sig_tail, axis = 0))
            var_tail += var_shift
            var_head -= var_shift
            #var_head = np.trace(np.cov(sig_head))
            cut_ratio = var_head/var_tail
            if verbose:
                print("# of discarded cells:", idis.size)
                print("# current data dimension:", self.get_size())
                print("Cut ratio:", cut_ratio)

        # is it necessary to perform PCA or I can just simply calculate the covariance?
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


    # -------------------------ica group--------------------------
    def ica_cell_rank(self, cell_group= 10, n_components = 4):
        '''
        given that the data is already cleaned. Perform ica to evaluate the individual components
        '''
        selected_signal = self.signal[:,cell_group]
        dff_ica, a_mix, s_mean = ica_sorting.ica_dff(selected_signal, n_comp = n_components)
        a_n2 = a_mix**2
        cell_ranks = np.argsort(a_n2, axis = 0) # importance of cells in each ic
        n2_coeffs = a_n2.sum(axis = 0)
        return dff_ica, cell_ranks, n2_coeffs



# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    raw_fname = global_datapath+'Jun13_A1_GCDA/'
    raw_data = np.load(raw_fname + 'ultra_cleaned.npz')
    ppl = pipeline(raw_data, raw = False)
    cell_group = np.arange(1000)
    n_ica = 10
    dff_ica, a_mix, s_mean = ppl.ica_cell_rank(cell_group, n_components = n_ica)
    NT = dff_ica.shape[0]
    print(dff_ica.shape,a_mix.shape)
    dff_origin = ppl.get_cells_index(cell_group)[0][:1500]
    figr = stat_present.dff_rasterplot(dff_origin,dt = 0.5, fw = 7.0)
    figr.savefig(raw_fname + 'raster_800')
    fig_ica = plt.figure(figsize = (6,4))
    ax = fig_ica.add_subplot(111)
    ax.plot(0.5*np.arange(NT)/60., dff_ica+np.arange(n_ica)*0.1)
    ax.set_xlabel('Time (min)', fontsize = 12)
    plt.tight_layout()
    fig_ica.savefig(raw_fname + 'ica_'+str(n_ica))




if __name__ == '__main__':
    main()
