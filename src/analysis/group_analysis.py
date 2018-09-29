"""
A general class for multiple dataset analysis.
"""

import numpy as np
from single_analysis import grinder
import os
import glob
from collections import deque

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/'
global_datapath_portable= '/media/sillycat/DanData/'


class mass_grinder(object):
    '''
    This is a class dealing with a group of fish instead of a single fish.
    '''

    def __init__(self, work_path, name_flag = 'lb'):
        '''
        initialize the class, check the workpath
        '''
        self._parse_folder_(work_path, name_flag)
        self.n_fish = 0 # the number of fish
        self.keys = deque()
        self.grinder_arr = deque()

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
i                   print("Added fish:", fish_key)



    def cleanup(self):
        '''
        clean up the class of the old data.
        '''

