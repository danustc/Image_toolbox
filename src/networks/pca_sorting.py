'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA, FastICA



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


def ica_dff(dff_data, n_comp = 4):
    '''
    directly use the ICA algorithm in sklearn
    '''
    dff_std = dff_data/dff_data.std(axis = 0)# standardize data
    ica = FastICA(n_components = n_comp, max_iter = 300)
    dff_recon = ica.fit_transform(dff_std) # reconstruct signals
    a_mix = ica.mixing_ # the estimated mixing matrix
    return dff_recon, a_mix


def group_varsplit(dff_data, arg_sv, var_contrast = 0.95):
    '''
    dff_data: the DF/F data of all the cells
    arg_sv: the indices of cells ranked by importance
    var_contrast: the variance of the first group should cover how much percentage of the overall variance?
    This is a test algorithm, function is not guaranteed.
    '''
    NT, NP = dff_data.shape

    cov_sum = np.cov(dff_data[:,arg_sv].T)
    cov_diag = np.diag(cov_sum) # the diagonal elements of the covariance matrix 
    cov_accumu = np.cumsum(cov_diag)  # the accumulated covariance.
    cut_ind = np.searchsorted(cov_accumu/cov_accumu[-1], var_contrast)
    print("cut_ind:", cut_ind)

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
        self.n_group = n_group
        self.groups = np.array_split(self.data, n_group, axis = 1)
        self.cum_count = np.array([sgroup.shape[1] for sgroup in self.groups]).cumsum() #cumulative count in the subgroups
        self.cum_count -= self.cum_count[0]


    def subgroup_pca(self, verbose = True):
        '''
        Perform PCA on each of the subgroups only once
        '''
        sub_var = np.sqrt(self.gvar) #the variance countability in subgroups 
        ind_discard = [] # the indices of discarded cells
        ind_preserv = []
        for dff_sub in self.groups:
            # perform PCA
            CT, V = pca_raw(dff_sub, sub_var)
            if verbose:
                print("The number of principle components that covers the majority of variance:", V.shape[0])
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

        return ipre, idis
#----------------------------------------------Test the function-------------------------------------------

def main():
    '''
    Test written on 04/30/2017.
    '''

    CT, V = pca_raw(dff_raw, var_cut = 0.95)
    print(V.shape)
    fig = stat_present.PCA_trajectory_matrix(CT, dim_select = [0,1,2,3,4])
    fig.savefig(global_datapath+'pc_test_habe')
    figv = stat_present.pc_component_grid(V[:,:100],npc = 55)
    figv.savefig(global_datapath+'pc_celldistri')
    a_sorted = cell_sorting(V)

    #independent component analysis
    dff_recon, a_mix = ica_dff(dff_raw[:, a_sorted[:70]],n_comp = 3)
    figi = stat_present.ic_plot(dff_recon)
    figi.savefig(global_datapath+'ica_test_70')

if __name__ == '__main__':
    main()


