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
import src.networks.pca_sorting as pca_sorting
import src.networks.ica_sorting as ica_sorting
import matplotlib.pyplot as plt
import h5py

from sklearn import linear_model
from sklearn.cluster import KMeans
from collections import deque

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
    def __init__(self, data_file, dt=0.5):
        self._coord = None
        self._signal = None
        self.dt = dt
        self.parse_data(data_file)

    def parse_data(self, data_file):
        try:
            self.hf = h5py.File(data_file, 'r+')
            try:
                self.coord = np.array(self.hf['coord'])
                self.signal = np.array(self.hf['signal'])
            except KeyError:
                print("No signal data stored in the file!")
                sys.exit(1)
        except OSError:
            print("Unable to open the file.")
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
    def ica_clustering(self, c_fraction = 0.40, n_components = 3, n_clusters = 4):
        '''
        given that the data is already cleaned. Perform ica to evaluate the individual components
        clustering the neurons based on their IC coefficients
        '''
        NT, NC = self.get_size()
        n_select = int(NC*c_fraction)  # number of neurons that are used for ica calculation
        selected_signal = self.signal[:,:n_select]
        dff_ica, a_mix, s_mean = ica_sorting.ica_dff(selected_signal, n_comp = n_components)

        lr = linear_model.LinearRegression()
        lr.fit(dff_ica, self.signal[:, n_select:])
        ico_total = np.r_[a_mix, lr.coef_] # the ic coefficients of all the extracted neurons 
        cls_pred = KMeans(n_clusters, random_state = None).fit(ico_total)
        labels = cls_pred.labels_
        #------- Next, let's divide the data into the groups based on the clustering labels.
        signal_clustered = deque()
        coord_clustered = deque()
        a_mix_clustered = deque()
        for ll in range(n_clusters):
            idx = np.where(labels == ll)[0]
            signal_clustered.append(self.signal[:,idx])
            coord_clustered.append(self.coord[idx])
            a_mix_clustered.append(idx)

        self.ic = dff_ica
        self.ic_coefs = ico_total
        self.ic_label = a_mix_clustered
        return coord_clustered, signal_clustered

    def clean_up(self):
        self.hf.close()

# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    n_ica = 3
    n_clu = 4
    cf = 0.40
    local_datafolder = 'FB_resting_15min'
    full_path = global_datapath + local_datafolder
    data_list = glob.glob(full_path + '/*dff.h5')

    for df_name in data_list:
        basename = '_'.join(os.path.basename(df_name).split('.')[0].split('_')[:3])
        print(basename)
        PL = pipeline(df_name)
        coord_clustered, signal_clustered = PL.ica_clustering(c_fraction = cf, n_components = n_ica, n_clusters = n_clu)
        fig_ic = stat_present.ic_plot(PL.ic, dt = 0.5, title = basename)
        fig_ic.savefig(full_path + '/'+ basename+ '_ic')
        plt.close(fig_ic)
        fig_cl = stat_present.cluster_dimplot(PL.ic_coefs, PL.ic_label)
        fig_cl.savefig(full_path + '/' + basename + '_icdim')
        plt.close(fig_cl)
        PL.clean_up()

        for nc in range(n_clu):
        #    sub_hf = h5py.File(full_path + '/' + basename + '_cl_'+ str(nc) + '.h5', 'w')
        #    sub_hf.create_dataset('coord', data = coord_clustered[nc])
        #    sub_hf.create_dataset('signal', data = signal_clustered[nc])
        #    sub_hf.close()
            cl_coord = coord_clustered[nc]
            cl_signal = signal_clustered[nc]
            print("# of cells:", cl_signal.shape[1])
            sub_dset = dict()
            sub_dset['coord'] = cl_coord
            sub_dset['signal'] = cl_signal
            np.savez(full_path + '/' + basename + '_cl_' + str(nc), **sub_dset)
            NC = cl_coord.shape[0]
            if NC>200:
                n_trunc = 200
            else:
                n_trunc = None
            fig_r = signal_plot.dff_rasterplot(cl_signal, n_truncate = n_trunc)
            fig_r.savefig(full_path + '/' + basename + '_raster_' + str(nc) )
            plt.close(fig_r)



if __name__ == '__main__':
    main()
