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
dp_list = ['Jul2017/', 'Aug2018/homo/', 'Aug2018/het/']


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

    def __init__(self):
        '''
        initialize the class, check the workpath
        '''
        self.n_fish = 0 # the number of fish
        self.keys = deque() # This array stores fish names and acquisition dates
        self.grinder_arr = deque() # This queue creates a a series of grinder analysers for the fish loaded
        self.folder_group = deque() # folder group. helps tracing back the fish groups 


    def parse_folder(self, work_path, nf = 'lb', shared_info = None):
        '''
        Parsing the folder. Should contain .npz files
        Warning: this does not wipe off the old data inside the class. One has to manually clean up the class if you want everything reset.
        info: the extra information of the folder, can be genotypes.
        '''
        flist = glob.glob(work_path + '*'+ nf+ '*.npz')
        basename = os.path.basename(work_path)
        nfl =  len(flist)
        if nfl>0:
            self.folder_group.append(nfl + self.n_fish) # the new group: where to start
            sni = str(self.n_fish)
            for fname in flist:
                fish_grinder = grinder()
                sta = fish_grinder.parse_data(fname, info = shared_info)
                if sta:
                    fish_key = fish_grinder.basename
                    self.keys.append(fish_key)
                    self.grinder_arr.append(fish_grinder)
                    self.n_fish +=1
                    print("Added fish:", fish_key)

            snf = str(self.n_fish)


    def activity_calc(self):
        '''
        calculate the activity levels of all the neurons in each cell.
        '''
        for g in self.grinder_arr:
            g.activity_sorting(sort = False)
            print("Finished calculating activities of the fish:", g.basename)



    def mask_coverage_statistics(self):
        '''
        Update on 10/02/2018: This is really clumsy. How can I improve it?
        Make a statistics for the masks covered in the group.
        Construct a DataFrame with mask name as indices and fish name as keys
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
            # just give me statistics of all the masks covered
            gm = self.group_mask
        else:
            gm = n_mask
            # gm = self.group_mask[n_mask]

        try:
            NF = len(self.fish_anno)
        except NameError:
            print("self.fish_anno is not defined.")
            return

        # These will be used as the column indices of the data frame.
        NG = len(gm)
        mname = np.genfromtxt(global_datapath_ubn + mask_database, dtype = 'str', delimiter = '\t') # load the name of masks
        fish_names = [self.keys[nn] for nn in self.fish_anno]
        fish_infos = [self.grinder_arr[nn].info for nn in self.fish_anno]
        mask_abvs = [mask_abbreviation(mn) for mn in mname[gm]] # mask abbreviations
        mask_mean = np.zeros((NF, NG))
        mask_sem = np.zeros((NF, NG))


            # iterate over selected masks
        for fa, ii in zip(self.fish_anno, range(NF)):
            for mm, jj in zip(gm, range(NG)):
                # iterate over annotated fish
                cind = self.grinder_arr[fa].select_mask(mm)
                if len(cind) > 0:
                    act_m = self.grinder_arr[fa].stat[cind, -1]
                    mask_mean[ii, jj] = act_m.mean()
                    mask_sem[ii,jj] = stats.sem(act_m)

        double_inds = [fish_names, fish_infos]
        activity_stat = DF(mask_mean, index = double_inds, columns = mask_abvs)

        return activity_stat



    def cleanup(self):
        '''
        clean up the class of the old data.
        '''
        self.keys = deque()
        self.grinder_arr = deque()


# ----------------------------Below is a test main function. -------------

def main():


    path_WT = global_datapath_ubn + FB_resting_folder + dp_list[0]
    path_homo = global_datapath_ubn + FB_resting_folder + dp_list[1]
    path_het = global_datapath_ubn + FB_resting_folder + dp_list[2]
    MG = mass_grinder()
    #MG.parse_folder(path_WT, shared_info = 'WT')
    MG.parse_folder(path_homo, shared_info = 'homo')
    MG.parse_folder(path_het, shared_info = 'het')
    MG.activity_calc()
    m_sum = MG.mask_coverage_statistics()
    a_stat = MG.mask_activity_statistics()
    print(a_stat)
    ax = anview.label_scatter(m_sum)
    plt.show()
    mask_mean = m_sum.mean(axis = 1)
    mask_se = m_sum.std(axis = 1)

if __name__ == '__main__':
    main()
