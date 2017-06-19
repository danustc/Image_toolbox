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

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    '''
    def __init__(self, data, raw = True):
        self._data = None
        self.data = data
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

    def dff_munging(self, fw = 4, nt = 10):
        '''
        calculate the df_f over the whole data set and save it
        '''
        self.signal = df_f.dff_raw(self.signal, ft_width = fw, ntruncate = nt)[0]

    def pca_layered_sorting(self,var_cut = 0.99, shuffle = True, verbose = True):
        '''
        perform the layered pca sorting on the data and shrink the size
        '''
        if shuffle:
            self.shuffle_data()

        NT, NP = self.get_size()
        gpca = pca_sorting.group_pca(self.signal, gvar = var_cut)

        while(NP > NT):
            gpca.group_division()
            ipre, idis = gpca.subgroup_pca(verbose)
            print(ipre.size)
            self.dff_cleaning(ipre) # after dff_cleaning, self.data is updated. 
            if shuffle:
                self.shuffle_data()
            NT, NP = self.get_size()
            if verbose:
                print("# of discarded cells:", idis.size)
                print("# current data dimension:", self.get_size())
            gpca.data = self.signal

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
    raw_fname = global_datapath+'Jun13_A1_GCDA/'
    raw_data = np.load(raw_fname + 'merged.npz')
    ppl = pipeline(raw_data)
    ppl.pca_layered_sorting(var_cut = 0.95)
    ppl.display_select(np.arange(10), raw_fname + 'Mostactive_10')
    ppl.display_select(np.arange(-10, 0), raw_fname + 'Leastactive_10')
    ppl.display_raster(ndisp = np.arange(500),figpath = raw_fname + 'raster')



if __name__ == '__main__':
    main()
