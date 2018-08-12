"""
A general class for analysis, which can be overloaded by many modules.
"""
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
global_datapath = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/'
from df_f import dff_AB

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

    def spatial_gridding(self, ng = (2, 2, 2) ):
        '''
        classify cells into different small categories using spatial classification
        rev_coord: if true, the coordinates are arranged in z-y-x; otherwise, x-y-z.
        Warning: the order of coord and ng must be consistent, i.e., both z-y-x or both x-y-z.
        output: an array of raveled indices in the order of z-y-x.
        '''
        coord = self.coord
        H, edges = np.histogramdd(coord, bins = ng) # 3D histogram
        print("Number of bins:", H.shape)

        if self.rev:
            cz, cy, cx = coord[:,0], coord[:,1], coord[:,2]
            egz, egy, egx = edges
            bz, by, bx = ng
        else:
            mx, my, mz = coord.max(axis = 0)
            cx, cy, cz = coord[:,0], coord[:,1], coord[:,2]
            egx, egy, egz = edges
            bx, by, bz = ng

        egx[0] -=1.0e-06
        egy[0] -=1.0e-06
        egz[0] -=1.0e-06

        # find the indices of edges for each neuron
        ind_x = np.searchsorted(egx, cx) - 1
        ind_y = np.searchsorted(egy, cy) - 1
        ind_z = np.searchsorted(egz, cz) - 1
        print(ind_x.min(), ind_y.min(), ind_z.min())

        rav_label = np.ravel_multi_index((ind_z, ind_y, ind_x), (bz, by, bx))
        self.group_mark = rav_label #divide cells into groups
        return rav_label

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


    def shutDown(self):
        pass


def coregen():
    date_folder = 'Jun07_2018/'
    grinder_core = grinder()
    data_path = global_datapath+ date_folder+'Jun07_2018_B5_dff.npz'
    grinder_core.parse_data(data_path)
    #grinder_core.activity_sorting()
    return grinder_core



if __name__ == '__main__':
    coregen()
