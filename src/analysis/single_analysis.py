"""
A general class for single-dataset analysis, which can be overloaded by many modules.
"""
import numpy as np
from scipy.stats import norm
from scipy.signal import savgol_filter
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
from df_f import dff_AB
from src.shared_funcs.numeric_funcs import gaussian1d_fit
import src.visualization.signal_plot as signal_plot
import clustering
global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Aug2018/'

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
                except KeyError:
                    print('No signal data stored in the file!')
                    return False
                if 'annotation' in data_pz.keys():
                    self.annotated = True
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

    def activity_sorting(self, nbin = 40, sort = True):
        '''
        Inference of the datapoints belonging to peaks and calculate level and standard deviation of the background.
        '''
        NT, NC = self.NT, self.NC
        activity_map = np.zeros([NT, NC], dtype = 'bool')
        ms = np.zeros([NC,3]) # baseline mean, noise, integral
        sig_integ = np.zeros(NC) # signal integral
        for nf in range(NC):
            cell_signal = self.signal[:,nf]
            sig_ind, background, noi = dff_AB(cell_signal, gam = 0.02, nbins = nbin)
            activity_map[sig_ind, nf] = True # set the activity map to True
            sig_integ = (cell_signal-background).sum()
            ms[nf] = np.array([background, noi, sig_integ]) # mean and std
            if nf%100 ==0:
                print("Finished calculating activity of ", nf, "cells.")

        if sort:
            integ_ind = np.argsort(ms[:,2])[::-1] # descending order
            self._trim_data_(integ_ind) # OK this is not a very elegant way to put it. 
            self.stat = ms[integ_ind]
            self.activity = activity_map[:, integ_ind] # sort the activity map as well
        else: # leave it unsorted
            self.stat = ms
            self.activity = activity_map


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


    def cutoff_bayesian(self, PH_const = 1., nb = 200, stake = 0.95, activity_range = 0.95, conserve_cutting = False):
        '''
        Use Bayesian inference to set the cutoff value that divides active/inactive neurons.
        if PHE > stake, then we believe that H happens
        '''
        if self.stat is None:
            print("No statistics data.")
            return
        else:
            int_val = self.stat[:,-1]
            hist, be = np.histogram(int_val, bins = nb, density = True, range = (0., activity_range*int_val.max()))
            PE_hist = savgol_filter(hist, window_length = 5, polyorder = 3)
            ind_peak = np.argmax(PE_hist)
            beh = (be[1:] + be[:-1])*0.5 # take the center value of each bin
            mu = beh[ind_peak] # The center of Gaussian distribution
            hsig = beh[ind_peak] - beh[0] # the half-sigma of the Gaussian distribution
            PEH = norm.pdf(beh, loc = mu, scale = hsig)
            PHE = PEH * PH_const/(PE_hist+0.001)#arg_be
            PHE[PHE>1.] = 1.
            if conserve_cutting:
                # conservative cutting
                beh_cut = np.where(PHE < stake)[0][0] # where to cut in the probability density function x-axis
            else:
                # aggressive cutting
                beh_cut = np.where(PHE > stake)[0][-1]+1 # where to cut in the probability density function x-axis
            activity_cut = beh[beh_cut] # if a cell's activity falls below activity cut,
            print("cut_off activity:", beh_cut, activity_cut)
            num_cut = (int_val< activity_cut).sum() # How many cells have activity levels falling below the activity cut?
            print("rejected cells:", num_cut)
            self.n_cut = num_cut
            cut_position = np.array([activity_cut, num_cut])

            return PHE, beh, cut_position


    def cutoff_simple(self, nb = 200, conf_level = 0.95, activity_range= 0.95):
        '''
        use simple model of gaussian to infer where the cutoff line should be.
        Warning: This function is far from being complete.
        '''
        if self.stat is None:
            print("No statistics data.")
            return
        else:
            # create a Gaussian distribution
            int_val = self.stat[:,-1]
            hist, be = np.histogram(int_val, bins = nb, density = True, range = (0., activity_range*int_val.max()))
            PE_hist = savgol_filter(hist, window_length = 5, polyorder = 3)
            ind_peak = np.argmax(PE_hist)
            beh = (be[1:] + be[:-1])*0.5 # take the center value of each bin
            be_bg = beh[:2*ind_peak+1]
            PE_peak_left = PE_hist[:ind_peak+1]
            PE_bg = np.concatenate((PE_peak_left, PE_peak_left[::-1][1:]))

            print("Fit parameters:", mu, sigx)


    def shuffle_control(self, n_shuffle = 2000):
        '''
        shuffle the first n_shuffle data to create a new dataset.
        '''
        shuffle_sig = np.copy(self.signal[:,:n_shuffle]) # maybe this copy is not necessary? 
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

    def saveas(self, newpath = None):
        '''
        save the dataset as the new file.
        '''
        if newpath is None:
            newpath = global_datapath_ubn + self.basename + '_cleaned'
        cleaned_dataset = {'coord':self.coord, 'signal':self.signal}
        np.savez(newpath, **cleaned_dataset)


    def shutDown(self):
        pass



def main():
    '''
    some initial munging of the datasets.
    '''
    data_path = global_datapath_ubn + 'Aug23_2018_B4_ref.npz'
    grinder_core = grinder()
    grinder_core.parse_data(data_path)
    grinder_core.activity_sorting()
    grinder_core.background_suppress(sup_coef = 0.0)
    grinder_core.saveas()


if __name__ == '__main__':
    main()
