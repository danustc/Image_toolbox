'''
Created by Dan on 07/11/2017.
Pipeline for batch calculation of raw_f into Delta F/F.
Last update: 08/12/2018, removed simple var sorting.
'''
import sys
import os
import glob
import numpy as np
import h5py
from df_f import *
from munging import coord_edgeclean
import matplotlib.pyplot as plt

global_datapath_ubn  = '/home/sillycat/Programming/Python/data_test/'
portable_datapath = '/media/sillycat/DanData/'
package_path_win  = r"C:\Users/Admin/Documents/GitHub/Image_toolbox\\" # This is for windows
sys.path.append(package_path)
global_datapath_win = r"D:\/Data/2018-08-02\\"

# ---------------------Below are small functions for data cleaning -------------

class pipeline(object):
    '''
    pipeline of calculating df/f, smoothing and sorting the activity of the cells by variance
    Note: the coordinate arrangement in this class is always z-y-x instead of x-y-z. Conversion only occurs in downstream processings.
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

        self.stat = None

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

    def load_dff(self, dff_name):
        '''
        load df/f, do more processing if necessary
        '''
        dff_data = np.load(dff_name)
        self.coord = dff_data['coord']
        self.signal = dff_data['signal']
        self.rawf = None


    def _list_properties_(self):
        '''
        List all the properties (that can be saved)
        '''
        pass
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
        if self.rawf is not None:
            self.rawf = self.rawf[:,ind_trim]

    def time_truncate(self, t_trunc = 10):
        '''
        trim the first t_trunc seconds.
        '''
        n_trunc = int(t_trunc/self.dt)
        self.signal = self.signal[n_trunc:]
        self.rawf = self.rawf[n_trunc:]


    def dff_calc(self, ft_width = 6, filt = True):
        '''
        I have to add one more step to validate the baseline.
        '''
        dffr = dff_raw(self.rawf, ft_width, ntruncate = 30)
        if filt:
            self.signal = dff_expfilt_group(dffr, self.dt, 1.8)
        else:
            self.signal = dffr

    def baseline_cleaning(self, bcut = 100.0):
        min_raw = self.rawf.min(axis = 0)
        invalid_ind = min_raw < bcut
        print("Fake cells:", invalid_ind.sum())
        self._trim_data_(~invalid_ind)

    def valid_check(self, df_th = 7.):
        '''
        remove cells that have dffs larger than df_th.
        '''
        invalid = np.any(self.signal > df_th, axis = 0)
        print("# of invalid cells:", invalid.sum())
        self.signal = self.signal[:, ~invalid]
        self.coord = self.coord[~invalid]


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


    def frequency_representation(self, N_cut = 2000, tw = 300, kt = 10, kf = 0.3, cstd = False, save_file = None):
        '''
        calculate frequency domain representation: spectrogram
        '''
        if cstd:
            signal = (self.signal-self.stat[:,0])/self.stat[:,1]
        else:
            signal = self.signal

        if np.isscalar(N_cut):
            fq_ind = np.arange(N_cut)
        elif len(N_cut) ==2:
            fq_ind = np.arange(N_cut[0], N_cut[1])
        else:
            fq_ind = N_cut
        n_freq = len(fq_ind)
        sg_temp = []
        sg_aver = []
        for nf in fq_ind:
            cell_signal = signal[:, nf] # takeout single spectrums
            spgram, k_max = spec.spectrogram(cell_signal,self.dt, tw, kt, kf)
            sg_temp.append(spgram)
            sg_aver.append((spgram**2).sum(axis = 1))

        sg_temp = np.stack(sg_temp)
        sg_aver = np.stack(sg_aver)
        if save_file is not None:
            spec_gram = {'temp': sg_temp, 'aver': sg_aver}
            np.savez(save_file, **spec_gram)
        return sg_temp, sg_aver


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
        copile a dictinary and save it
        '''
        data = {'signal':self.signal, 'coord':self.coord}

        np.savez(save_path, **data)
#------------------------------The main test function ---------------------

def main_rawf():
    #data_folder = 'FB_resting_15min/Jul2017/'
    #data_folder = 'FB_resting_15min/Aug02_2018/'
    raw_list = glob.glob(global_datapath_win +'*_merged.npz')
    #raw_list = glob.glob(portable_datapath+'Jul*merged.npz')
    for raw_file in raw_list:
        acquisition_date = '_'.join(os.path.basename(raw_file).split('.')[0].split('_')[:-1])
        raw_data = np.load(raw_file)
        ppl = pipeline(raw_data)
        #ppl.location_cleaning()
        ppl.edge_truncate(edge_width = 2.0)
        ppl.baseline_cleaning(bcut = 160.0)
        ppl.dff_calc(ft_width = 6, filt = True)
        ppl.valid_check(df_th = 6.)
        ppl.save_cleaned_dff(global_datapath_win + acquisition_date + '_dff')
        print("Finished processing:", acquisition_date)

def main_dff():
    data_folder = 'FB_resting_15min/Jun07_2018/'
    dff_list = glob.glob(global_datapath_ubn+data_folder+'*dff.npz')
    ppl = pipeline()
    for dff_file in dff_list:
        acquisition_date = '_'.join(os.path.basename(dff_file).split('.')[0].split('_')[:-1])
        ppl.load_dff(dff_file)
        ppl.save_cleaned_dff(dff_file)
        sg_file = global_datapath_ubn + data_folder + acquisition_date + '_sg'
        sg_temp, sg_aver = ppl.frequency_representation(N_cut = 5000, tw = 300, kt = 10, kf = 0.25, save_file = sg_file)
        print("Finished processing:", acquisition_date)



if __name__ == '__main__':
    main_rawf()
    #main_dff()
