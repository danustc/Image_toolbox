"""
A general class for multiple dataset analysis.
"""

import numpy as np
from single_analysis import grinder
import os
import glob
from collections import deque
import matplotlib.pyplot as plt

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/'
global_datapath_portable= '/media/sillycat/DanData/'
FB_resting_folder = 'FB_resting_15min/'
mask_database = 'mask_name.txt'
dp_list = ['Jul2017/', 'Aug2018/']

class mass_grinder(object):
    '''
    This is a class dealing with a group of fish instead of a single fish.
    '''

    def __init__(self, work_path, name_flag = 'lb'):
        '''
        initialize the class, check the workpath
        '''
        self.n_fish = 0 # the number of fish
        self.keys = deque()
        self.grinder_arr = deque()

        self.parse_folder(work_path, name_flag)

    def parse_folder(self, work_path, nf):
        '''
        Parsing the folder. Should contain .npz files
        Warning: this does not wipe off the old data inside the class. One has to manually clean up the class if you want everything reset.
        '''
        flist = glob.glob(work_path + '*'+ nf+ '*.npz')
        nfl =  len(flist)
        if nfl>0:
            for fname in flist:
                fish_grinder = grinder()
                sta = fish_grinder.parse_data(fname)
                if sta:
                    fish_key = os.path.basename(fname).split('.')[0]
                    self.keys.append(fish_key)
                    self.grinder_arr.append(fish_grinder)
                    self.n_fish +=1
                    print("Added fish:", fish_key)

    def select_fish(self, NF, fish_key):
        pass


    def anatomical_mask_statistics(self):
        '''
        Make a statistics for the masks covered in the group.
        '''
        nanno = []
        group_mask = np.array([])
        for ii in range(self.n_fish):
            g = self.grinder_arr[ii]
            if g.annotated:
                group_mask = np.union1d(group_mask, g.keys) # This is naturally sorted
                nanno.append(ii)

        ngm = group_mask.size # The total number of mask covered
        nfm = len(nanno)
        mask_summary = np.zeros((ngm, nfm))

        for jj in range(nfm):
            ind_fish = nanno[jj]
            key_fish = self.grinder_arr[ind_fish].keys
            kst_fish = self.grinder_arr[ind_fish].key_stat
            ik =  np.searchsorted(group_mask, key_fish)

            mask_summary[ik,jj] = kst_fish

        return group_mask, mask_summary



    def cleanup(self):
        '''
        clean up the class of the old data.
        '''
        self.keys = deque()
        self.grinder_arr = deque()


# ----------------------------Below is a test main function. -------------

def main():
    mname = np.genfromtxt(global_datapath_ubn + mask_database, dtype = 'str', delimiter = '\t') # load the name of masks


    path_Jul2017 = global_datapath_ubn + FB_resting_folder + dp_list[0]
    path_Aug2018 = global_datapath_ubn + FB_resting_folder + dp_list[1]
    MG = mass_grinder(path_Aug2018)
    g_mask, m_sum = MG.anatomical_mask_statistics()

    mask_covered = mname[g_mask]

    mask_mean = m_sum.mean(axis = 1)
    mask_se = m_sum.std(axis = 1)
    plt.plot(g_mask, m_sum.sum(axis = 1), 'x')
    plt.show()

if __name__ == '__main__':
    main()
