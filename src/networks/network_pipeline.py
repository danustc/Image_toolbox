'''
A small pipeline for network analysis. Added by Dan on 06/18/2017.
Last update: 07/27/2017.
'''
print(__doc__)
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import glob
import numpy as np
from src.shared_funcs.tifffunc import read_tiff
import src.visualization.stat_present as stat_present
import src.visualization.signal_plot as signal_plot
import src.networks.clustering as clustering
import src.networks.pca_sorting as pca_sorting
import src.networks.ica_sorting as ica_sorting
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
            self._trim_data_(ipre) # after trim_data, self.signal is updated. 
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
        the mixing matrix is returned
        '''
        selected_signal = self.signal[:,cell_group]
        dff_ica, a_mix, s_mean = ica_sorting.ica_dff(selected_signal, n_comp = n_components)
        return dff_ica, a_mix




# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    n_ica = 3
    n_active = 20
    n_group = 6
    n_raster = 200
    folder_list = glob.glob(global_datapath + '*GCDA*')
    for work_folder in folder_list:
        raw_dff = np.load(work_folder + '/merged_dff.npz')
        ppl = pipeline(raw_dff)
        coord_mostactive = ppl.coord[:n_active,:] # The coordinates of the most active cells
        print(coord_mostactive)
        sc_most = np.max(ppl.signal[:,0])/4.0
        fign = signal_plot.nature_style_dffplot(ppl.signal[:,:n_active], dt = 0.5, sc_bar = sc_most)
        fign.savefig(work_folder+ '/most_active_'+ str(n_active))
        plt.close()

        fign = signal_plot.nature_style_dffplot(ppl.signal[:,-n_active:], dt = 0.5, sc_bar = 0.10)
        fign.savefig(work_folder+ '/least_active_'+ str(n_active))
        plt.close()
        # raster-plot the most active cells and do ica
        for ix in range(n_group):
            cell_group = np.arange(n_raster)+ix*n_raster
            #dff_ica, a_mix = ppl.ica_cell_rank(cell_group, n_components = n_ica)
            dff_ica, a_mix, s_mean = ica_sorting.ica_dff(ppl.signal[:,cell_group], n_comp = n_ica)
            NT = dff_ica.shape[0]
            dff_origin = ppl.get_cells_index(cell_group)[0]
            figr = signal_plot.dff_rasterplot(dff_origin,dt = 0.5, fw = 7.0)
            figr.savefig(work_folder+ '/raster_' + 'g'+ str(ix)+ '_'+ str(n_raster))
            plt.close()
            fig_ica = stat_present.ic_plot(dff_ica, dt = 0.5)
            fig_ica.savefig(work_folder + '/ica_'+str(n_ica) + '_g' + str(ix))
            plt.close()

            # clustering of a_mix
            figc, R, Z= clustering.dis2cluster(a_mix, p_levels = 4, yield_z = True)
            figc.savefig(work_folder+'/ic_cluster_'+str(n_ica)+ '-g' + str(ix))

            ind_list_L = clustering.subtree(Z, n_raster, 'L', True)
            ind_list_R = clustering.subtree(Z, n_raster, 'R', True)
            cluster_indices = [ind_list_L, ind_list_R]
            figl = stat_present.cluster_dimplot(a_mix, cluster_indices, 'rg')
            figl.savefig(work_folder+ '/ic_cluster_scatter'+ '-g'+str(ix))


if __name__ == '__main__':
    main()
