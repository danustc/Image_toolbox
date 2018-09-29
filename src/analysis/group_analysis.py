"""
A general class for multiple dataset analysis.
"""

import numpy as np
from single_analysis import grinder
import os
import glob
from collections import deque

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/'
global_datapath_portable= '/media/sillycat/DanData/'

dp_list = ['Jul2017/', 'Aug2018']

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

        self._parse_folder_(work_path, name_flag)

    def _parse_folder_(self, work_path, nf):
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
        nanno = 0
        group_mask = np.array([])
        for g in self.grinder_arr:
            if g.annotated:
                group_mask = np.union1d(group_mask, g.keys)
                nanno +=1

        ngm = group_mask.size # The total number of mask covered



    def cleanup(self):
        '''
        clean up the class of the old data.
        '''
        self.keys = deque()
        self.grinder_arr = deque()


# ----------------------------Below is a test main function. -------------

def main():
    path_Jul2017 = global_datapath_ubn + dp_list[0]
    MG = mass_grinder(path_Jul2017)


if __name__ == '__main__':
    main()
