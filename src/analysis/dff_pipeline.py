'''
Created by Dan on 07/11/2017.
Pipeline for batch calculation of raw_f into Delta F/F.
Last update: 11/04/2017, replacing npz with hdf5.
'''
import sys
import os
import glob
import numpy as np
import h5py
from df_f import *
from munging import coord_edgeclean
import simple_variance as simple_variance

global_datapath_ubn  = '/home/sillycat/Programming/Python/data_test/'
portable_datapath = '/media/sillycat/DanData/'
package_path_win  = r"C:\Users/Admin/Documents/GitHub/Image_toolbox\\" # This is for windows
sys.path.append(package_path)
#global_datapath_win = r"D:\/Data/2018-06-07\\"

# ---------------------Below are small functions for data cleaning -------------

class pipeline(object):
    '''
    pipeline of calculating df/f, smoothing and sorting the activity of the cells by variance
    '''
    def __init__(self, raw_data = None, dt = 0.5):
        self._coord = None
        self._rawf = None
        self._signal = None
        self.dt = dt
        if raw_data is None:
            pass
        elif self.parse_data(raw_data):
            self.dff_calc()


    def parse_data(self, raw_data):
        try:
            self.coord = raw_data['coord']
            try:
                self.rawf = raw_data['data']
            except KeyError:
                print('No raw key value stored in the file! ')
                sys.exit(1)
        except KeyError:
            print('The data has wrong format.')
            self.coord = None
            self.rawf = None
            self.signal = None
            sys.exit(1)

        print("raw data loaded.")
        return True

    def load_dff(self, dff_data):
        keys = dff_data.keys()


    # ----------------------Below are the property members ----------------------
    @property
    def rawf(self):
        return self._rawf
    @rawf.setter
    def rawf(self, new_rawf):
        self._rawf = new_rawf

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
    # ---------------------Below are the analyzing functions-------------------

    def get_size(self):
        return self.signal.shape

    def _trim_data_(self, ind_trim ):
        self.signal = self.signal[:,ind_trim]
        self.coord = self.coord[ind_trim,:]
        self.rawf = self.rawf[:,ind_trim]

    def time_truncate(self, t_trunc = 10):
        '''
        trim the first t_trunc seconds.
        '''
        n_trunc = int(t_trunc/self.dt)
        self.signal = self.signal[n_trunc:]
        self.rawf = self.rawf[n_trunc:]


    def dff_calc(self, ft_width = 6, filt = True):
        dffr = dff_raw(self.rawf, ft_width, ntruncate = 25)
        if filt:
            self.signal = dff_expfilt_group(dffr, self.dt, 1.8)
        else:
            self.signal = dffr

    def svar_sorting(self, var_cut = 0.95):
        '''
        simple variance-based sorting
        '''
        crank, dvar = simple_variance.simvar_global_sort(self.signal)
        sum_var = np.cumsum(dvar)
        sum_var /= sum_var[-1]
        n_cut = np.searchsorted(sum_var, var_cut)
        self._trim_data_(crank[:n_cut])

    def zscore_sorting(self, zs = 0.05):
        '''
        calculate Z-score of each neuron and remove ones with Z-score below zs.
        '''
        NT, NC = self.get_size()
        hist_sorted = np.zeros([NT, NC])
        for nf in range(NC):
            dff_hist(self.signal[:,nf], rect = False, noise_norm = True)



    def edge_truncate(self, edge_width = 10.0, verbose = True):
        '''
        Warning: this is for unlabeled raw-fluorescence dataset, in which the coordinates are ordered z-y-x instead of x-y-z.
        '''
        coord = self.coord
        px_max = np.max(coord, axis = 0)
        print(px_max)
        if verbose:
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
        self.rawf = np.delete(self.rawf, it_discard, axis = 1)
        if verbose:
            print("Removed", it_discard.size, "edge neurons")
            print("Final data dimension:", self.get_size())


    def save_cleaned_h5(self, save_path):
        '''
        compile a dictinary and save it
        '''
        fo = h5py.File(save_path, 'w')
        fo.create_dataset('coord', data = self.coord)
        fo.create_dataset('signal', data = self.signal)
        fo.close()


    def save_cleaned_dff(self, save_path):
        '''
        compile a dictinary and save it
        '''
        data = {'signal':self.signal, 'coord':self.coord}
        np.savez(save_path, **data)
#------------------------------The main test function ---------------------

def main():
    data_folder = 'FB_resting_15min/'
    raw_list = glob.glob(global_datapath_ubn+data_folder+'Jul2017/*merged.npz')
    #raw_list = glob.glob(portable_datapath+'Jul*merged.npz')
    #data_folder = 'FMR1/'
    for raw_file in raw_list:
        acquisition_date = '_'.join(os.path.basename(raw_file).split('.')[0].split('_')[:-1])
        raw_data = np.load(raw_file)
        ppl = pipeline(raw_data)
        ppl.edge_truncate(edge_width = 7.0)
        ppl.dff_calc(ft_width = 6, filt = True)
        ppl.save_cleaned_dff(global_datapath_ubn  +data_folder+ acquisition_date + '_dff')
        print("Finished processing:", acquisition_date)


if __name__ == '__main__':
    main()
