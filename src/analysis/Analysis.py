"""
A general class for analysis, which can be overloaded by many modules.
"""
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
global_datapath = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/'
portable_datapath = '/media/sillycat/DanData/'
from df_f import dff_AB
import src.algos.spectral_clustering as sc
import src.visualization.signal_plot as signal_plot
import clustering
import matplotlib.pyplot as plt

class grinder(object):
    '''
    the general data grinder module, which reads signal, coordinates and do shuffling when necessary.
    '''
    def __init__(self,coord = None, signal = None, rev = True):
        self.signal = signal
        self.coord = coord
        self.rev = rev # whether the coordinates are reversed.
        print("Loaded.")
        self._get_size_()
        self.group_mark = -1*np.ones(self.NC) #all the cells are uncategorized

        self.stat = None
        self.activity = None

    def _trim_data_(self, ind_trim):
        self.signal = self.signal[:,ind_trim]
        self.coord = self.coord[ind_trim,:]

    def _get_size_(self):
        '''
        Explicitly save the data size so that we don't have to calculate the array dimensions everytime.
        '''
        if self.signal is not None:
            self.NT, self.NC = self.signal.shape
        else:
            self.NT, self.NC = 0, 0

    def parse_data(self, data_file, rev = True):
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
                print(data_file)
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

        self.rev = rev
        self._get_size_()

    def rev_coords(self):
        '''
        reverse the coordinate orders from z-y-x to x-y-z.
        '''
        self.coord = self.coord[:,::-1]
        self.rev = not(self.rev)

    def activity_sorting(self, nbin = 40):
        '''
        Inference of the datapoints belonging to peaks and calculate level and standard deviation of the background.
        '''
        NT, NC = self.NT, self.NC
        activity_map = np.zeros([NT, NC], dtype = 'bool')
        ms = np.zeros([NC,3]) # baseline mean, noise, integral
        sig_integ = np.zeros(NC) # signal integral
        for nf in range(NC):
            cell_signal = self.signal[:,nf]
            sig_ind, background, noi = dff_AB(cell_signal, gam = 0.05, nbins = nbin)
            activity_map[sig_ind, nf] = True # set the activity map to True
            sig_integ = (cell_signal-background).sum()
            ms[nf] = np.array([background, noi, sig_integ]) # mean and std

        integ_ind = np.argsort(ms[:,2])[::-1] # descending order
        self._trim_data_(integ_ind) # OK this is not a very elegant way to put it. 
        self.stat = ms[integ_ind]
        self.activity = activity_map[:, integ_ind] # sort the activity map as well

    def background_suppress(self, sup_coef = 0.010, shuffle = True):
        '''
        suppress background. Information needed: activity map
        '''
        NC = self.NC
        activity, signal = self.activity, self.signal
        for nf in range(NC):
            act = activity[:, nf]
            cell_signal = signal[:,nf]
            sig_A = cell_signal[act] # the signal data points
            sig_B = cell_signal[~act] # the background data points
            SNR = sig_A.mean()/sig_B.std() # the definition of SNR in images
            sfac = 1.-np.exp(-sup_coef*SNR**2)
            print("SNR:", SNR, "suppress fraction:", sfac)
            self.signal[~act, nf] *= sfac # suppress the background
            if shuffle:
                '''
                shuffle the background points
                '''
                NB = sig_B.size
                shuff_order = np.random.permutation(np.arange(NB))
                b_ind = np.where(~act)[0]
                self.signal[b_ind, nf] = self.signal[b_ind[shuff_order], nf]


    def digitization(self):
        '''
        set all the active points to 1 and inactive points to 0.
        '''
        NC = self.NC
        activity = self.activity
        dig_signal = np.zeros_like(self.signal)
        for nf in range(NC):
            act = activity[:,nf]
            dig_signal[act,nf] = 1

        return dig_signal

    def normalization(self):
        '''
        normalize the amplitude of all the neurons with mean.
        '''
        NC = self.NC
        activity = self.activity
        norm_signal = np.zeros_like(self.signal)
        for nf in range(NC):
            act = activity[:, nf]
            mean_sig = self.signal[act, nf].mean()
            norm_signal[:, nf] = self.signal[:,nf]/mean_sig

        return norm_signal

    def shutDown(self):
        pass


def coregen():
    date_folder = 'Aug02_2018/'
    grinder_core = grinder()
    #data_path = global_datapath+ date_folder+'Aug02_2018_B2_dff.npz'
    data_path = portable_datapath + 'Jul19_2017_A2_dff.npz'
    grinder_core.parse_data(data_path)
    grinder_core.activity_sorting()
    plt.hist(grinder_core.activity, bins = 200)
    plt.show()
    grinder_core.background_suppress(sup_coef = 0.0001)
    N_cut = 1500
    th = 0.23
    signal_test = grinder_core.signal[10:,:N_cut]
    fig_all = signal_plot.dff_rasterplot(signal_test, fw = (7.0,4.5))
    fig_all.savefig('all_'+str(N_cut))
    W = sc.corr_afinity(grinder_core.signal[10:,:N_cut], thresh = th, kill_diag = False)
    L = sc.laplacian(W)
    w, v = sc.sc_unnormalized(L, n_cluster = 20)

    n_clu = 5
    y_labels = clustering.spec_cluster(grinder_core.signal[:,:N_cut],n_clu, threshold = th)
    #return grinder_core
    sig_clusters = np.zeros([1775, n_clu])
    leg_cluster = []
    sub_cluster = []
    for nl in range(n_clu):
        signal_nl = signal_test[:, y_labels == nl]
        NT, NC = signal_nl.shape
        if NC > 500:
            sub_cluster.append(nl)
        sig_clusters[:,nl] = signal_nl.mean(axis = 1)
        fig = signal_plot.dff_rasterplot(signal_nl)
        fig.savefig('cluster_'+str(nl), dpi = 200)
        fig.clf()
        leg_cluster.append('cluster_'+str(nl+1))


    fig_mean = signal_plot.compact_dffplot(sig_clusters, dt = 0.5, sc_bar = 0.20, tbar = 2)
    fig_mean.savefig('cluster_mean', dpi = 200)


    nl = sub_cluster[0]
    sub_population = signal_test[:,y_labels ==nl]
    W = sc.corr_afinity(signal_test[:,y_labels == nl], thresh = 0.95*th)
    L = sc.laplacian(W)
    w, v = sc.sc_unnormalized(L, n_cluster = 20)
    plt.close('all')
    plt.plot(w, '-x')
    plt.show()
    n_clu = int(input("the # of clusters in the subset:") )

    y_labels = clustering.spec_cluster(sub_population, n_clu, threshold = th)
    sig_clusters = np.zeros([1775, n_clu])
    for nl in range(n_clu):
        signal_nl = sub_population[:, y_labels == nl]
        NT, NC = signal_nl.shape
        sig_clusters[:,nl] = signal_nl.mean(axis = 1)
        fig = signal_plot.dff_rasterplot(signal_nl)
        fig.savefig('subcluster_'+str(nl), dpi = 200)
        fig.clf()


    fig_mean = signal_plot.compact_dffplot(sig_clusters, dt = 0.5, sc_bar = 0.20, tbar = 2)
    fig_mean.savefig('sub_mean', dpi = 200)


if __name__ == '__main__':
    coregen()
