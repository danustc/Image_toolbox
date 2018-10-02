"""
A general class for multiple dataset analysis.
"""

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import src.visualization.anatomy_view as anview
import numpy as np
from single_analysis import grinder
import os
import glob
from collections import deque
from pandas import DataFrame as DF
from scipy import stats
import matplotlib.pyplot as plt

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/'
global_datapath_portable= '/media/sillycat/DanData/'
FB_resting_folder = 'FB_resting_15min/'
mask_database = 'mask_name.txt'
dp_list = ['Jul2017/', 'Aug2018/']

def mask_abbreviation(m_name):
    '''
    Create an abbreviation of the input mask name.
    '''
    mn_div = m_name.split('-')
    mn_main = mn_div[0][:4]
    mn_sub = mn_div[1]
    if len(mn_sub)>0: # the secondary part is not empty
        sub_list = mn_sub.split(' ')
        if len(sub_list) == 1:
            mask_abrv = '-'.join([mn_main, sub_list[0][:5]])
        else: # the sublist has multiple parts
            if sub_list[-1].isdigit(): # the last part is digit?
                num_mark = sub_list[-1]
                sub_abbrev = '-'.join([spart[:3] for spart in sub_list[:-1]])
                sub_abbrev += num_mark
            else:
                sub_abbrev = '-'.join([spart[:3] for spart in sub_list])

            mask_abrv = ''.join([mn_main, sub_abbrev])

        return mask_abrv

    else: # only the main name exists
        return mn_main



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

    def parse_folder(self, work_path, nf = 'lb'):
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


    def activity_calc(self):
        '''
        calculate the activity levels of all the neurons in each cell.
        '''
        for g in self.grinder_arr:
            g.activity_sorting(sort = False)
            print("Finished calculating activities of the fish:". g.basename)



    def mask_coverage_statistics(self):
        '''
        Update on 10/02/2018: This is really clumsy. How can I improve it?
        Make a statistics for the masks covered in the group.
        Construct a DataFrame with mask name as indices and fish name as keys
        Also, if activity is calculated, calculate the mask average.
        '''
        nanno = []
        group_mask = np.array([])
        for ii in range(self.n_fish):
            g = self.grinder_arr[ii]
            if g.annotated:
                group_mask = np.union1d(group_mask, g.keys).astype('uint16') # This is naturally sorted
                nanno.append(ii)

        # group_mask: all the mask that have been covered

        ngm = group_mask.size # The total number of mask covered
        nfm = len(nanno)
        mask_summary = np.zeros((ngm, nfm))

        for jj in range(nfm):
            ind_fish = nanno[jj]
            key_fish = self.grinder_arr[ind_fish].keys
            kst_fish = self.grinder_arr[ind_fish].key_stat
            ik =  np.searchsorted(group_mask, key_fish)

            mask_summary[ik,jj] = kst_fish

        mname = np.genfromtxt(global_datapath_ubn + mask_database, dtype = 'str', delimiter = '\t') # load the name of masks
        col_names = [mask_abbreviation(mn) for mn in mname[group_mask]]
        row_inds = [self.keys[nn] for nn in nanno]
        df_mask = DF(data = mask_summary.T, index = row_inds, columns = col_names)
        self.group_mask = group_mask # keep this part of information
        self.fish_anno = nanno

        return df_mask


    def mask_activity_statistics(self, n_mask = None):
        '''
        calculate statistics of the masks
        Question: should I do this fish-wise or group-wise?
        if n_mask is None: calculate the statistics of all the covered masks; toherwise calculate that for the selected mask only.
        '''
        if n_mask is None:
            gm = self.group_mask
        else:
            gm = self.group_mask[n_mask]

        for mm in gm:
            # iterate over selected masks
            for fa in self.fish_anno:
                # iterate over annotated fish
                cind = self.grinder_arr[fa].select_mask(mm)
                act_m = self.grinder_arr[fa].stat[cind, -1]

                ma_m, sa_m = act_m.mean(), stats.sem(act_m) # mean and std



    def cleanup(self):
        '''
        clean up the class of the old data.
        '''
        self.keys = deque()
        self.grinder_arr = deque()


# ----------------------------Below is a test main function. -------------

def main():


    path_Jul2017 = global_datapath_ubn + FB_resting_folder + dp_list[0]
    path_Aug2018 = global_datapath_ubn + FB_resting_folder + dp_list[1]
    MG = mass_grinder(path_Jul2017)
    MG.parse_folder(path_Aug2018)
    MG.activity_calc()
    g_mask, m_sum = MG.anatomical_mask_statistics()

    ax = anview.label_scatter(m_sum)
    plt.show()
    mask_mean = m_sum.mean(axis = 1)
    mask_se = m_sum.std(axis = 1)

if __name__ == '__main__':
    main()
