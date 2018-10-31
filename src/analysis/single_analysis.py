"""
A general class for single-dataset analysis, which can be overloaded by many modules.
"""
import numpy as np
from scipy.signal import savgol_filter
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
import glob
from df_f import dff_AB, activity_level
from src.shared_funcs.numeric_funcs import gaussian1d_fit
from scipy.stats import norm, kruskal, mannwhitneyu# two non-parametric tests
from scipy.interpolate import interp1d
import src.visualization.signal_plot as signal_plot
import clustering
import matplotlib.pyplot as plt
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Jul2017/Annotated/'

def bool2str(bool_arr):
    '''
    convert a boolean array into a 0-1 string.
    '''
    int_arr = bool_arr.astype('uint8')
    str_arr = ''.join([str(ii) for ii in int_arr])

    return str_arr

class grinder(object):
    '''
    the general data grinder module, which reads signal, coordinates and do shuffling when necessary.
    '''
    def __init__(self,coord = None, signal = None, rev = True, dt = 0.50):
        self.signal = signal
        self.coord = coord
        self.rev = rev # whether the coordinates are reversed.
        if coord is not None:
            print("Raw data Loaded.")
            self._get_size_()
            self.group_mark = -1*np.ones(self.NC) #all the cells are uncategorized

        self.stat = None
        self.activity = None
        self.annotated = False
        self.dt = dt

    def _trim_data_(self, ind_trim):
        self.signal = self.signal[:,ind_trim]
        self.coord = self.coord[ind_trim,:]
        if self.annotated:
            self.neuron_label = self.neuron_label[ind_trim, :]

    def _get_size_(self):
        '''
        Explicitly save the data size so that we don't have to calculate the array dimensions everytime.
        '''
        if self.signal is not None:
            self.NT, self.NC = self.signal.shape
        else:
            self.NT, self.NC = 0, 0

    def parse_data(self, data_file, rev = True, info = None):
        basename, fmt = os.path.basename(data_file).split('.')
        self.basename = basename
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
                    return False

            except OSError:
                print("Unable to open the file.")
                return False

        else:
            try:
                print(data_file)
                data_pz = np.load(data_file)
                try:
                    self.coord = data_pz['coord']
                    self.signal = data_pz['signal']
                    self._get_size_()
                except KeyError:
                    print('No signal data stored in the file!')
                    return False
                if 'annotation' in data_pz.keys():
                    self.annotated = True
                    print("The dataset is annotated.")
                    self.keys = data_pz['annotation'][-1] # the masks that have been covered 
                    self.neuron_label = data_pz['annotation'][:-1] # the labeling of each neuron
                    self.key_stat = self.neuron_label.sum(axis = 0) # the total number of neurons in that mask
                else:
                    self.annotated = False

            except OSError:
                print("Unable to open the file.")
                return False

        self.info = info
        self.rev = rev
        self._get_size_()

        return True

    def rev_coords(self):
        '''
        reverse the coordinate orders from z-y-x to x-y-z.
        '''
        self.coord = self.coord[:,::-1]
        self.rev = not(self.rev)



    def activity_sorting(self, nbin = 40, sort = True, bg_shuffling = False, upsampling = 1):
        '''
        Inference of the datapoints belonging to peaks and calculate level and standard deviation of the background.
        '''
        NT, NC = self.NT, self.NC
        activity_map = np.zeros([NT, NC], dtype = 'bool')
        ms = np.zeros([NC,3]) # baseline mean, noise, integral
        sig_integ = np.zeros(NC) # signal integral
        tt = np.arange(NT)
        for nf in range(NC):
            cell_signal = self.signal[:,nf]
            sig_ind, background, noi = dff_AB(cell_signal, nbins = nbin)
            activity_map[sig_ind, nf] = True # set the activity map to True

            if bg_shuffling: # shuffle the background data points
                bg_ind = np.arange(NT)[~activity_map[:, nf]]
                cell_signal[bg_ind] = np.random.permutation(cell_signal[bg_ind])

            if upsampling ==1:
                sig_integ = (cell_signal-background).sum()
            else: # do an upsampling first
                #f = interp1d(tt, cell_signal)
                #cell_interp = f(tmid) # interpolated value
                sig_integ = activity_level(cell_signal, background, tt, upsampling)

            ms[nf] = np.array([background, noi, sig_integ]) # mean and std

            #if nf%100 ==0:
                #print("Finished calculating activity of ", nf, "cells.")

        if sort:
            integ_ind = np.argsort(ms[:,2])[::-1] # descending order
            self._trim_data_(integ_ind) # OK this is not a very elegant way to put it. 
            self.stat = ms[integ_ind]
            self.activity = activity_map[:, integ_ind] # sort the activity map as well
        else: # leave it unsorted
            self.stat = ms
            self.activity = activity_map



    def acceptance_directcompare(self, cont_min = 4, alpha = 0.001):
        '''
        directly compare two groups of data
        '''
        if self.activity is None:
            print("the activity map is not calculated.")
            return

        activity, signal = self.activity, self.signal
        mwu = np.zeros([self.NC, 2]) # mann-whitney U test result
        kru = np.zeros([self.NC, 2]) # kruskal test result
        print("# of neurons:", self.NC)
        for nf in range(self.NC):
            act_map = activity[:,nf]
            baseline = self.stat[nf, 0]
            s = bool2str(act_map)
            cont_s = s.split('0')

            len_cont = np.array([len(ct) for ct in cont_s])

            if len_cont.max() > cont_min:
                cell_sig = signal[:,nf]
                X1 = cell_sig[act_map] - baseline
                #X2 = cell_sig[cell_sig > baseline]
                #X2 = cell_sig[~act_map]
                X2 = cell_sig
                mwu[nf] = mannwhitneyu(X1, X2, alternative = 'greater')
                kru[nf] = kruskal(X1, X2)
            else:
                print("neuron %d : fail"%nf)
                print("No continuous peak is detected.")
                mwu[nf, 1] = 1.
                kru[nf, 1] = 1.

        acceptance = np.logical_and(mwu[:,1] < alpha, kru[:,1] < alpha)
        #acceptance = kru[:,1] < alpha

        return acceptance

    def select_mask(self, n_mask):
        '''
        return the average activity of neurons within a mask only.
        '''
        if self.annotated:
            m_included = (self.keys == n_mask) # check whether the key is included in the mask.
            if np.any(m_included):
                ind_mask = np.where(m_included)[0] #This is the index of the mask in self.keys
                mask_coverage = self.neuron_label[:, ind_mask] # the neuronal labeling of n_mask
                cell_ind = np.where(mask_coverage)[0] # these are the indices of the cells that belong to mask # n_mask.
                return cell_ind

            else:
                print("The mask", n_mask, "is not covered.")
                return []
        else:
            print("The fish has not been annotated yet.")
            return []


    def shuffled_activity(self, N_times, ind_shuf, upsampling = 2):
        '''
        calculate the shuffled activity: 1 time or N times (NC x N_times array)
        '''
        tt = np.arange(self.NT)
        NS = len(ind_shuf)
        if N_times ==1:
            activity_control = np.zeros(NS)
            shuff_sig = self.shuffle_control(ind_shuf) # NT x NC
            '''
            I need to rewrite the whole section.
            '''
            for cc in range(NS):
                id_peak, baseline, _ = dff_AB(shuff_sig[:,cc] )
                #baseline = self.stat[cc,0]
                activity_control[cc] = activity_level(shuff_sig[:,cc], baseline, tt, upsampling)

            return activity_control


        else: # shuffle multiple times
            activity_control = np.empty((NS, N_times))
            baseline_sums = np.zeros_like(activity_control)
            for ii in range(N_times): # iterate through N_times shuffling
                shuff_sig = self.shuffle_control(ind_shuf)
                for cc in range(NS):
                    id_peak, baseline, _ = dff_AB(shuff_sig[:,cc])
                    #baseline = self.stat[cc,0]
                    baseline_sums[cc, ii] = baseline
                    activity_control[cc, ii] = activity_level(shuff_sig[:,cc], baseline, tt, upsampling)

            return activity_control, baseline_sums


    def cutoff_shuffling(self, N_times = 10, conf_level = 3.0, a_thresh = 10.0):
        '''
        Check whether the detected signal is activity or noise.
        since I am just comparing one value with a set of values, there is no statistical test method required.
        '''
        if self.stat is None:
            print("The activity has not been calculated.")
            return

        a_real = self.stat[:,-1] # the activity of real time traces
        b_real = self.stat[:,0]
        NS = np.where(a_real < a_thresh)[0][0]
        print("The cutoff position:", NS)
        a_control, b_sums = self.shuffled_activity(N_times, np.arange(NS, self.NC), upsampling = 2)
        amean, astd = a_control.mean(axis = 1), a_control.std(axis = 1)
        bmean, bstd = b_sums.mean(axis = 1), b_sums.std(axis = 1)
        acceptance = np.zeros(self.NC).astype('bool') # whether or not should I accept each cell?
        acceptance[:NS] = True

        '''
        Now, let's do some expensive but more reliable test
        '''
        for cc in range(self.NC-NS):
            X1 = a_real[cc+NS] # test data set 1
            Y1 = b_real[cc+NS]
            mx, stx = amean[cc], astd[cc]
            my, sty = bmean[cc], bstd[cc]

            #if X1 > m2 + conf_level*st2 or X1 > a_thresh:
            if X1 > mx + conf_level*stx and Y1 < my -conf_level*sty:
                print("accept cell ", cc+NS)
                print("activity: ", X1, mx, stx)
                print("baseline: ",  Y1, my, sty)
                print("-------------------------------")
                acceptance[cc+NS] = True

            else:
                print("reject cell ", cc+NS)
                print("activity:", X1, mx, stx)

        #ac_mean, ac_std = a_control.mean(axis = 1), a_control.std(axis = 1) # The mean and std of the 
        #acceptance = a_real > (ac_mean + conf_level*ac_std)
        return acceptance


        # 10/14/2018: I removed the cutoff_simple function.
        # 10/17/2018: I removed the cutoff_Bayesian function.
    def shuffle_control(self, ind_shuffle = None):
        '''
        shuffle the first n_shuffle data to create a new dataset.
        '''
        if ind_shuffle is None:
            shuffle_sig = np.copy(self.signal) # maybe this copy is not necessary? 

        else:
            shuffle_sig = np.copy(self.signal[:,ind_shuffle])

        shuffle_sig = np.random.permutation(shuffle_sig) # only shuffles along the 0th dimension, which is convenient for me! :D
        return shuffle_sig


    def activity_evolution(self, twindow = 300):
        '''
        twindow: time window for integrating the signal intensity
        prerequisite: self.stat is not None.
        '''
        signal = self.signal
        nw = int(twindow//self.dt) # the width of the sliding window in the unit of time stamps
        w_filter = np.ones(nw)
        tw_act = np.empty([self.NT-nw+1, self.NC])
        for ii in range(self.NC):
            tw_act[:,ii] = np.convolve(signal[:,ii], w_filter, mode = 'valid')

        return tw_act


    def background_suppress(self, sup_coef = 0.010, shuffle = True, verbose = False):
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
            if len(sig_A) > 0:
                SNR = sig_A.mean()/sig_B.std() # the definition of SNR in images
            else:
                SNR = 0.
            sfac = 1.-np.exp(-sup_coef*SNR**2)
            if verbose:
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

    def saveas(self, newpath = None):
        '''
        save the dataset as the new file.
        '''
        if newpath is None:
            newpath = global_datapath_ubn + self.basename + '_cleaned'
        cleaned_dataset = {'coord':self.coord, 'signal':self.signal, 'annotation':self.neuron_label, 'masks':self.keys}
        np.savez(newpath, **cleaned_dataset)


    def shutDown(self):
        pass

# ----------------------------------- Here are some test functions. -------------------------------------------

def main():
    '''
    some initial munging of the datasets.
    '''
    data_list  = glob.glob(global_datapath_ubn + '*_ref_lb.npz')
    grinder_core = grinder()

    for data_path in data_list:
        grinder_core.parse_data(data_path)
        grinder_core.activity_sorting(bg_shuffling = False, upsampling = 1)
        acceptance = grinder_core.acceptance_directcompare()
        print(acceptance.sum(), " neurons accepted.")

        ind = np.arange(grinder_core.NC)[acceptance]
        ind_comp = np.arange(grinder_core.NC)[~acceptance]
        grinder_core.background_suppress(sup_coef = 0.0)
        #grinder_core._trim_data_(acceptance)
        grinder_core.saveas()


if __name__ == '__main__':
    main()
