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
import src.networks.noise_removal as noise_removal
import matplotlib.pyplot as plt

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    '''
    def __init__(self, data, raw = True, dt=0.5):
        self._data = None
        self.data = data
        self.dt = dt
        if raw:
            self.dff_munging()

    def _parse_data_(self):
        try:
            self.coord = self._data['coord']
            self.signal= self._data['data']
        except KeyError:
            print('Wrong data!')
            self.data = None

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


    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._parse_data_()

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
            print("# of discarded cells:", idis.size)
            print("# current data dimension:", self.get_size())
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
        self.shuffle_data(ind_shuffle = a_sorted)


    def dff_cleaning(self, ipreserve):
        '''
        clean the dataset by sorting the preserved indices
        '''
        signals = self.signal[:,ipreserve]
        coords = self.coord[ipreserve, :] # select coordinates and signals
        new_data = dict()
        new_data['coord'] = coords
        new_data['data'] = signals
        self.data = new_data #renewed data

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
    raw_fname = global_datapath+'Jun13_B2_control/'
    raw_data = np.load(raw_fname + 'merged.npz')
    ppl = pipeline(raw_data)
    ppl.pca_layered_sorting(var_cut = 0.95)
    ppl.display_select(np.arange(10), raw_fname + 'Mostactive_10')
    ppl.display_select(np.arange(-10, 0), raw_fname + 'Leastactive_10')
    ppl.display_raster(ndisp = np.arange(500),figpath = raw_fname + 'raster')

    plt.clf()
    plt.plot(ppl.wd)
    plt.savefig(raw_fname+'wd')


if __name__ == '__main__':
    main()
