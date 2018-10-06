'''
A small pipeline for network analysis. Added by Dan on 06/18/2017.
Last update: 12/13/2017.
'''
print(__doc__)
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import glob
import numpy as np
from src.visualization import stat_present, signal_plot
import src.analysis.munging as munging
import src.analysis.pca_sorting as pca_sorting
import src.analysis.ica_sorting as ica_sorting
import src.analysis.clustering as clustering
import matplotlib.pyplot as plt
import h5py

from sklearn import linear_model
from sklearn.cluster import KMeans
from collections import deque

global_datapath = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/'

def npztoh5(fpath, direct = 'f', new_path = None):
    '''
    conversion between npz and h5 files.
    '''
    if direct == 'f':
        # convert from npz to h5
        try:
            data_set = np.load(fpath)
        except IOError:
            print("Wrong data path. Cannot open the file.")
            return

        k_list = data_set.keys()
        if new_path is None:
            base_name = fpath.split('.')[0]
            output_name = base_name + '.h5'
        else:
            output_name = new_path + '.h5'
        hf = h5py.File(output_name, 'w')
        for key in k_list:
            hf.create_dataset(key, data = data_set[key])

        hf.close()




class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    1. Do the sorting/cleaning if necessary.
    2. Do the detailed analysis if necessary.
    Conventions:    a. self.signal stores signals (either F or \Delta F/F) of all the extracted cells.
                    b. self.cood stores the coordinates (in micron) of all the extracted cells. If coord has two columns, the 0th is y and the 1st is x; if coord has three columns, the order is z,y,x (instead of x,y,z, to be consistent with Python array dimension orders)
    '''
    def __init__(self):
        self._coord = None
        self._signal = None
        self.dt = 0.5

    def parse_data(self, data_file):
        fmt = os.path.basename(data_file).split('.')[-1]
        if fmt == 'h5':
            try:
                print(data_file)
                hf = h5py.File(data_file, 'r+')
                try:
                    self.coord = np.array(hf['coord'])
                    self.signal = np.array(hf['signal'])
                    hf.close()
                except KeyError:
                    print("No signal data stored in the file!")
                    sys.exit(1)
            except OSError:
                print("Unable to open the file.")
                sys.exit(1)
        else:
            try:
                data_pz = np.load(data_file)
                try:
                    self.coord = data_pz['coord']
                    self.signal = data_pz['signal']
                except KeyError:
                    print('No signal data stored in the file!')
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

    def _trim_data_(self, idx , mode = 0):
        '''
        trim the data and keep the cells with indices idx only.
        '''
        if mode == 0:
            self.signal = self.signal[:,idx]
            self.coord = self.coord[idx, :]
        else: # delete the cells with indices idx
            self.signal = np.delete(self.signal,idx, 1) # delete columns from signal
            self.coord = np.delete(self.coord,idx, 0) # delete columns from signal

    def reorder_data(self, ind_shuffle = None):
        '''
        shuffle the signal orders and the coordinate orders so that the position is not biasing the groupwise pca output.
        '''
        if ind_shuffle is None:
            NT, NP = self.get_size()
            ind_shuffle = np.arange(NP)
            np.random.shuffle(ind_shuffle)

        self._trim_data_(ind_shuffle)


    def pca_layered_sorting(self,var_cut = 0.999, shuffle = True, verbose = True):
        '''
        perform the layered pca sorting on the data and shrink the size
        '''
        if shuffle:
            self.reorder_data()

        NT, NP = self.get_size()
        var_ratio = var_cut/(1.-var_cut)
        print("var ratio:",var_ratio)
        gpca = pca_sorting.group_pca(self.signal, gvar = var_cut)
        gpca.group_division()
        ipre, idis = gpca.subgroup_pca(fine_sort = True)

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
                self.reorder_data()
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
        self.reorder_data(ind_shuffle = a_sorted) #rank cell by the final activities
        if verbose:
            print("Final data dimension:", self.get_size())
        # end of pca_layered sorting 

    def save_cleaned(self, save_path, rev_zyx = True):
        '''
        Compile a dictionary and save it
        '''
        cleaned_data = dict()
        if rev_zyx:
            cleaned_data['coord'] = self.coord[::-1]
        else:
            cleaned_data['coord'] = self.coord
        cleaned_data['signal'] = self.signal
        np.savez(save_path, **cleaned_data)


    # -------------------------ica group--------------------------
    def ica_clustering(self, c_fraction = 0.40, n_components = 3, n_clusters = 3):
        '''
        given that the data is already cleaned. Perform ica to evaluate the individual components
        clustering the neurons based on their IC coefficients
        '''
        NT, NC = self.get_size()
        if c_fraction is None:
            selected_signal = self.signal
        else:
            n_select = int(NC*c_fraction)  # number of neurons that are used for ica calculation
            selected_signal = self.signal[:,:n_select]

        print(selected_signal.shape)
        dff_ica, a_mix, s_mean = ica_sorting.ica_dff(selected_signal, n_comp = n_components)
        print(a_mix.shape)

        lr = linear_model.LinearRegression()
        lr.fit(dff_ica, self.signal[:, n_select:])
        ico_total = np.r_[a_mix, lr.coef_] # the ic coefficients of all the extracted neurons 
        cls_pred = KMeans(n_clusters, random_state = None).fit(ico_total)
        labels = cls_pred.labels_
        print(labels.shape)
        km_centers = cls_pred.cluster_centers_
        #------- Next, let's divide the data into the groups based on the clustering labels.
        signal_clustered = deque()
        coord_clustered = deque()
        idx_clustered = deque()
        for ll in range(n_clusters):
            idx = np.where(labels == ll)[0]
            signal_clustered.append(self.signal[:,idx])
            coord_clustered.append(self.coord[idx])
            idx_clustered.append(idx)

        self.ic = dff_ica
        self.ic_coefs = ico_total
        self.clu_label = idx_clustered
        self.km_centers = km_centers
        return coord_clustered, signal_clustered

    def ica_interactive_cleaning(self):
        '''
        perform interactive cleaning on the raw_data
        '''
        mk =0
        NI = len(self.clu_label)
        ic_select = input("Please select the cluster to be removed:")
        print("Selected mode:", ic_select)
        ic_list = ic_select.split(',')
        for ic in ic_list:
            try:
                idx = self.clu_label[int(ic)] # find the indices to be deleted 
                self._trim_data_(idx, mode = 1)
                mk+=1
            except IndexError:
                print("Index Error.")
        return mk

    def stimuli_sort(self, trigger_signal, nbins = 50, n_iter = 1, n_sample = 5):
        '''
        sort the signals based on their correlation with the trigger.
        generate a histogram of the correlation function
        '''
        NT, NC = self.get_size()
        ntri = len(trigger_signal)
        if (NT > ntri):
            dff = self.signal[:ntri]
        elif(NT < ntri):
            dff = self.signal
            trigger_signal = trigger_signal[:NT] #truncate the longer one
        else:
            dff = self.signal
        trig_corr = np.empty(NC)
        for ii in range(NC):
            trig_corr[ii] = np.corrcoef(trigger_signal, dff[:,ii])[1,0]

        ind_sort = np.argsort(trig_corr)[::-1] # descending order
        self.reorder_data(ind_shuffle = ind_sort)
        if n_iter > 1:
            regressor = self.signal[:,:n_sample].mean(axis = 1)*trigger_signal
            dff = self.signal
            for ii in range(NC):
                trig_corr[ii] = np.corrcoef(regressor, dff[:,ii])[1,0]
            ind_sort = np.argsort(trig_corr)[::-1]
            self.reorder_data(ind_shuffle = ind_sort)
            return trig_corr[ind_sort], regressor
        else:
            return trig_corr[ind_sort]

    def clean_up(self):
        self.ic = None
        self.ic_coefs = None
        self.clu_label = None
        self.signal = None
        self.coord = None


    def __del__(self):
        # destructor test 
        print("The class dies.")

# --------------------------Below is the test section -------------------
def main():
    '''
    The test function of the pipeline.
    '''
    n_ica = 4
    n_clu = 5
    cf = 0.90
    local_datafolder = 'Aug02_2018/'
    full_path = global_datapath + local_datafolder
    data_list = glob.glob(full_path + '*B2*dff.npz')
    clean_list = []
    clean_fname = full_path+'clean_list.txt'

    PL = pipeline()
    for df_name in data_list:
        basename = '_'.join(os.path.basename(df_name).split('.')[0].split('_')[:-1])
        print("Processing file:", basename)
        PL.parse_data(df_name)
        # -------- PCA and ICA clustering ------------------

        # commented by Dan on 03/07
        PL.reorder_data(ind_shuffle = np.arange(1500))
        PL.signal = PL.signal[10:]
        #fig_sti = signal_plot.dff_rasterplot(PL.signal, n_truncate = ntr)
        #fig_sti.savefig(full_path + '/' + basename + '_sti')
        coord_clustered, signal_clustered = PL.ica_clustering(c_fraction = cf,n_components = n_ica, n_clusters = n_clu)

        fig_ica = stat_present.ic_plot(PL.ic)
        fig_ica.savefig(full_path + '/'+ basename + '_ic')
        plt.close(fig_ica)

        fig_cl = stat_present.cluster_dimplot(PL.ic_coefs, PL.clu_label)
        fig_cl.savefig(full_path + '/' + basename + '_icdim')
        plt.close(fig_cl)

        cluster_size = [len(cluster) for cluster in PL.clu_label]
        print("cluster sizes:", cluster_size)
        cluster_mean = []
        for nc in range(n_clu):
            cl_coord = coord_clustered[nc]
            cl_signal = signal_clustered[nc]
            print("# of cells:", cl_signal.shape[1])
            print("Cluster center:", PL.km_centers[nc])
            sub_dset = dict()
            sub_dset['coord'] = cl_coord[:,[2,1,0]] # this is for display. x,y,z ordered as x,y,z, so it is consistent with that in the image.
            sub_dset['signal'] = cl_signal
            np.savez(full_path + '/' + basename + '_cl_' + str(nc), **sub_dset)
            NC = cl_coord.shape[0]
            fig_r = signal_plot.dff_rasterplot(cl_signal, fw = (7.0, 5.0) )
            fig_r.savefig(full_path + '/' + basename + '_' + str(nc) )
            plt.close(fig_r)
            cluster_mean.append(cl_signal.mean(axis = 1))


        cluster_mean = np.array(cluster_mean).T
        fig_mean = signal_plot.compact_dffplot(cluster_mean, sc_bar = 0.2, tbar = 2 )
        fig_mean.savefig('ica_cluster_mean')
        #PL.save_cleaned(full_path+ '/'+ basename +'_cleaned',rev_zyx = False)
        #PL.save_cleaned(df_name,rev_zyx = False)
        #fig_r = signal_plot.dff_rasterplot(PL.signal, fw = 7.0, title = basename )
        #fig_r.savefig(full_path + '/' + basename + '_sti' )

        plt.close('all')





if __name__ == '__main__':
    main()
