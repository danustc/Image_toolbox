'''
A small pipeline for network analysis. Added by Dan on 06/18/2017.
Last update: 07/11/2017.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import numpy as np
from src.shared_funcs.tifffunc import read_tiff
import src.visualization.stat_present as stat_present
from src.visualization.brain_navigation import region_view
import src.networks.pca_sorting as pca_sorting
import src.networks.ica_sorting as ica_sorting
import src.networks.simple_variance as simple_variance
from src.networks.noise_removal import coord_edgeclean
import matplotlib.pyplot as plt

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
portable_datapath = '/media/sillycat/DanData/HQFB_redundancy_removed/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    1. Do the sorting/cleaning if necessary.
    2. Do the detailed analysis if necessary.
    Conventions:    a. self.signal stores signals (either F or \Delta F/F) of all the extracted cells.
                    b. self.cood stores the coordinates (in micron) of all the extracted cells. If coord has two columns, the 0th is y and the 1st is x; if coord has three columns, the order is z,y,x (instead of x,y,z, to be consistent with Python array dimension orders)
    '''
    def __init__(self, data, dt=0.5):
        self._coord = None
        self._signal = None
        self.dt = dt
        self.parse_data(data)

    def parse_data(self, data):
        try:
            self.coord = data['coord']
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

    def _trim_data_(self, ind_kept):
        '''
        trim the data and keep the cells with indices ind_kept only.
        '''
        self.signal = self.signal[:,ind_kept]
        self.coord = self.coord[ind_kept, :]

    def time_truncate(self, t_trunc = 10):
        '''
        trim the first t_trunc seconds.
        '''
        n_trunc = int(t_trunc/self.dt)
        self.signal = self.signal[n_trunc:]


    def svar_sorting(self, var_cut = 0.95):
        '''
        simple variance-based sorting
        '''
        crank, dvar = simple_variance.simvar_global_sort(self.signal)
        print(dvar)
        sum_var = np.cumsum(dvar)
        sum_var /= sum_var[-1]
        print(sum_var)
        n_cut = np.searchsorted(sum_var, var_cut)
        print(n_cut)
        self._trim_data_(crank[:n_cut])


    def shuffle_data(self, ind_shuffle = None):
        '''
        shuffle the signal orders and the coordinate orders so that the position is not biasing the groupwise pca output.
        '''
        if ind_shuffle is None:
            NT, NP = self.get_size()
            ind_shuffle = np.arange(NP)
            np.random.shuffle(ind_shuffle)

        self._trim_data_(ind_shuffle)


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
        # end of pca_layered sorting 


    def edge_truncate(self, edge_width = 10.0, verbose = True):
        # cut the edges of the dataset, edge_width unit: microns
        coord = self.coord
        px_max = np.max(coord, axis = 0)
        if verbose:
            print(px_max)
            print("Initial data dimension:", self.get_size())
        ix_left = coord_edgeclean(coord, edge_width, 'x', -1)
        ix_right = coord_edgeclean(coord, px_max[-1]-edge_width, 'x',1)
        iy_up = coord_edgeclean(coord, edge_width, 'y', -1)
        iy_down = coord_edgeclean(coord, px_max[1]-edge_width, 'y', 1)
        ix_discard = np.union1d(ix_left, ix_right)
        iy_discard = np.union1d(iy_up, iy_down)
        it_discard = np.union1d(ix_discard, iy_discard)
        self.coord = np.delete(self.coord, it_discard, axis = 0)
        self.signal = np.delete(self.signal, it_discard, axis = 1)
        if verbose:
            print("Removed", it_discard.size, "edge neurons")
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
    raw_fname = global_datapath+'HQ/Dec07_2016_B1/'
    raw_data = np.load(raw_fname + 'merged_dff.npz')
    ppl = pipeline(raw_data)
    print("Original data size:", ppl.get_size())
    ppl.svar_sorting(var_cut = 0.999)
    #ppl.time_truncate(10)
    print("Cleaned data size:", ppl.get_size())
    ppl.edge_truncate()
    n_ica = 5
    n_active = 20
    coord_mostactive = ppl.coord[:n_active,:]
    print(coord_mostactive)
    fign = stat_present.nature_style_dffplot(ppl.signal[:,:n_active], dt = 0.5, sc_bar = 0.5)
    fign.savefig(raw_fname + 'most_active_'+ str(n_active))
    plt.close()
    fign = stat_present.nature_style_dffplot(ppl.signal[:,-n_active:], dt = 0.5, sc_bar = 0.10)
    fign.savefig(raw_fname + 'least_active_'+ str(n_active))
    plt.close()
    # raster-plot the most active cells and do ica
    n_group = 5
    n_raster = 100
    for ix in range(n_group):
        cell_group = np.arange(n_raster)+ix*n_raster
        dff_ica, a_mix, s_mean = ppl.ica_cell_rank(cell_group, n_components = n_ica)
        NT = dff_ica.shape[0]
        dff_origin = ppl.get_cells_index(cell_group)[0]
        figr = stat_present.dff_rasterplot(dff_origin,dt = 0.5, fw = 7.0)
        figr.savefig(raw_fname + 'raster_' + 'g'+ str(ix)+ '_'+ str(n_raster))

        fig_ica = plt.figure(figsize = (6,4))
        ax = fig_ica.add_subplot(111)
        ax.plot(0.5*np.arange(NT)/60., dff_ica+np.arange(n_ica)*0.1)
        ax.set_xlabel('Time (min)', fontsize = 12)
        plt.tight_layout()
        fig_ica.savefig(raw_fname + 'ica_'+str(n_ica) + '_g' + str(ix))
        plt.close()




if __name__ == '__main__':
    main()
