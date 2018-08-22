'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from src.shared_funcs import tifffunc as tf
from collections import deque
from pca_funcs import *
import matplotlib.pyplot as plt
global_datapath = '/home/sillycat/Programming/Python/data_test/'

class group_pca(object):
    '''
    Perform PCA on a group of data with more columns on rows
    '''
    def __init__(self, raw_data = None, gvar= 0.95):
        '''
        Load data. Specify the variance cut-off.
        '''
        self._data = None
        self.gvar = gvar
        self.data= raw_data
        self.groups = None

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, new_data):
        self._data = new_data
        self.NT, self.NP = new_data.shape


    def group_division(self, n_group = None):
        '''
        separate the data into several groups
        '''
        if n_group is None:
            n_group = int(2*self.NP/self.NT)+1 # suggested divisions based on the data dimensions
            print("Group division:", n_group)
        self.n_group = n_group
        self.groups = np.array_split(self.data, n_group, axis = 1)
        self.cum_count = np.array([sgroup.shape[1] for sgroup in self.groups]).cumsum() #cumulative count in the subgroups
        self.cum_count -= self.cum_count[0]


    def subgroup_pca(self, fine_sort = True):
        '''
        Perform PCA on each of the subgroups only once
        '''
        sub_var = np.sqrt(self.gvar) #the variance countability in subgroups 
        ind_discard = [] # the indices of discarded cells
        ind_preserv = []
        for dff_sub in self.groups:
            # perform PCA
            CT, V = pca_raw(dff_sub, sub_var)
            a_sorted = cell_sorting(V)
            arg_head, arg_tail = group_varsplit(dff_sub, a_sorted, sub_var)
            ind_discard.append(arg_tail) # append the indices of cells to be discarded
            ind_preserv.append(arg_head)

        for igroup in range(self.n_group):
            #merge all the preserved and discarded indices
            ind_discard[igroup] += self.cum_count[igroup]
            ind_preserv[igroup] += self.cum_count[igroup]

        idis = np.concatenate(ind_discard)
        ipre = np.concatenate(ind_preserv)

        if fine_sort: # keep those cells that relatively have large variance in the to-be-discarded group
            data_residue = self.data[:,idis]
            CT, V = pca_raw(data_residue, self.gvar)
            a_sorted = cell_sorting(V)
            arg_head, arg_tail = group_varsplit(data_residue, a_sorted, self.gvar**2)
            idis_fine = idis[arg_tail]
            ipre_fine = np.concatenate((ipre, idis[arg_head]))
            return ipre_fine, idis_fine
        else:
            return ipre, idis
#----------------------------------------------Test the function-------------------------------------------

def main():
    local_datafolder = 'Jun_GCDA/'

if __name__ == '__main__':
    main()


