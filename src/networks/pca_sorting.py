'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA
import src.visualization.stat_present as stat_present
import src.visualization.brain_navigation as brain_navigation
from src.shared_funcs import tifffunc as tf
import matplotlib.pyplot as plt
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

def pca_dff(dff_data, n_comp = 3):
    '''
    0. load dff_data. Each column represents the df_f of a neuron.
    1. perform PCA on n_comp.
    2. visualization.
    '''
    pc_dff = PCA(n_components = n_comp)
    pc_dff.fit(dff_data)
    pc_trans = pc_dff.transform(dff_data)
    return pc_trans


# to understand PCA better, let's write a PCA from scratch (Yay! )
def pca_raw(data, var_cut = 0.95):
    '''
    data: uncentralized, unstandardlized data
    n_comp: the number of components to be extracted.
    var_cut: variance cut, ignore the contributions from minor principal components.
    '''
    N, P = data.shape # the number of measurements and the number of 'sensors'
    c_data = data-data.mean(axis = 0) # centralization
    U,s,V = np.linalg.svd(c_data) # svd 
    lam = s**2/(N-1)
    var_tot = lam.cumsum()/lam.sum() # 
    n_comp = np.searchsorted(var_tot, var_cut)+1 # the number of principal components that covers var_cut fraction of variance
    CT = U[:,:n_comp]*s[:n_comp] # the coefficients on the chosen PCs  
    V_signif = V[:n_comp]
    return CT, V_signif


# Next, try to remove the noisy cells which are not firin
def cell_sorting(V):
    '''
    classifying the cells into active and non-active groups according to their V coefficients. V contains the principal vectors sorted by the eigen values.
    returns the most and least active nselect neurons.
    '''
    NV, NP = V.shape
    sv = np.sum(-V**2, axis = 0)
    arg_sv = np.argsort(sv)
    return arg_sv



def group_varsplit(dff_data, arg_sv, var_contrast = 0.95):
    '''
    dff_data: the DF/F data of all the cells
    arg_sv: the indices of cells ranked by importance
    var_contrast: the variance of the first group should cover how much percentage of the overall variance?
    This is a test algorithm, function is not guaranteed.
    '''
    NT, NP = dff_data.shape

    cov_diag = np.var(dff_data[:,arg_sv], axis = 0)
    #cov_diag = np.diag(cov_sum) # the diagonal elements of the covariance matrix 
    cov_accumu = np.cumsum(cov_diag)  # the accumulated covariance.
    cut_ind = np.searchsorted(cov_accumu/cov_accumu[-1], var_contrast)

    arg_head = arg_sv[:cut_ind]
    arg_rear = arg_sv[cut_ind:]
    return arg_head, arg_rear


# --------------------------------# The class of group split 

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


