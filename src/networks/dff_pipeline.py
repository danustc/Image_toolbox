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
from noise_removal import coord_edgeclean
import simple_variance as simple_variance

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
portable_datapath = '/media/sillycat/DanData/HQFB_redundancy_removed/'
# ---------------------Below are small functions for data cleaning -------------

def raw2dff_clean(raw_loaded, dff_flag = 'dff', dt = 0.5, t_width = 1.5, saveraw = False):
    '''
    convert a raw fluorescence file into DF/F values.
    '''
    f_dic = dict(raw_loaded) # convert the Npz file into dictionary
    data_raw = f_dic.pop('data')
    dffr = dff_raw(data_raw, ft_width = 6, ntruncate = 20)
    if saveraw:
        f_dic['dff_raw'] = dffr
    f_dic['signal'] = dff_expfilt_group(dffr, dt, t_width)
    f_dic['t_features'] = np.array([dt, t_width]) # save the temporal information
    folder_path = os.path.dirname(raw_fname) +'/'
    dff_fname = os.path.basename(raw_fname).split('.')[0]+ '_' + dff_flag
    np.savez(folder_path + dff_fname, **f_dic)

# ---------------------Below is the main pipeline class

class pipeline(object):
    '''
    pipeline of calculating df/f, smoothing and sorting the activity of the cells by variance
    '''
    def __init__(self, raw_data, dt = 0.5):
        self._coord = None
        self._rawf = None
        self._signal = None
        self.dt = dt
        if self.parse_data(raw_data):
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
        dffr = dff_raw(self.rawf, ft_width, ntruncate = 20)
        if filt:
            self.signal = dff_expfilt_group(dffr, self.dt, 1.5)
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


    def edge_truncate(self, edge_width = 10.0, verbose = True):
        # cut the edges of the dataset, edge_width unit: microns
        coord = self.coord
        px_max = np.max(coord, axis = 0)
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


    def save_cleaned(self, save_path):
        '''
        compile a dictinary and save it
        '''
        fo = h5py.File(save_path, 'w')
        fo.create_dataset('coord', data = self.coord)
        fo.create_dataset('signal', data = self.signal)
        fo.close()

#------------------------------The main test function ---------------------

def main():
    #folder_list = glob.glob(portable_datapath+'May22*')
    data_folder = 'FB_resting_15min'
    data_list = glob.glob(global_datapath+data_folder + '/*.npz')
    for dset in data_list:
        basename = os.path.basename(dset)
        acquisition_date = '_'.join(basename.split('.')[0].split('_')[:3])
        print(acquisition_date)
        raw_data = np.load(dset)
        ppl = pipeline(raw_data)
        ppl.edge_truncate(edge_width = 5.0)
        ppl.dff_calc(ft_width = 6, filt = True)
        ppl.svar_sorting(var_cut = 0.99)
        ppl.save_cleaned(global_datapath + data_folder+'/'+ acquisition_date + '_merged_dff.h5')
        print("Finished processing:", acquisition_date)


if __name__ == '__main__':
    main()
