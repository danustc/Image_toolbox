'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA
from collections import deque

def _group_division_(NC, n_cut, n_tail = None):
    '''
    divide an array into several subgroups.
    '''
    if n_tail is None:
        n_tail = int(n_cut//2)

    n_res = np.mod(NC, n_cut) # residue
    n_slice = np.arange(0, NC, n_cut) + n_cut
    if n_res > n_tail or n_res ==0:
        return n_slice
    else:
        n_slice = n_slice[:-1] # crop the last element
        n_slice[-1] = NC
        return n_slice


def pca_dff(dff_data, n_comp = None, norm = True, cl_select = 1):
    '''
    0. load dff_data. Each column represents the df_f of a neuron.
    1. perform PCA on n_comp.
    2. visualization.
    This has consistent results with pca_raw. However, var_cut is not available.
    cl_select:
    '''
    NT, NC = dff_data.shape
    if n_comp is None:
        n_comp = NC
    pc_dff = PCA(n_components = n_comp)
    dff_mean = dff_data - dff_data.mean(axis = 0)
    if norm:
        stds = np.std(dff_mean, axis = 0) #
        dff_mean/= stds
    pc_dff.fit(dff_mean)
    pc_trans = pc_dff.transform(dff_mean)
    pc_vecs = pc_dff.components_
    eig_vals = pc_dff.explained_variance_
    if cl_select == 0:
        return pc_trans, pc_vecs, eig_vals
    else:
        q = NT/NC
        isq = 1./np.sqrt(q)
        if cl_select == 1: # select the components larger than the upper bound
            l_ubound = (1.+isq)**2
            n_clu = (eig_vals > l_ubound).sum() # take out all the eigenvalues that exceed marcenko-pastur distribution
        elif cl_select == -1:
            l_lbound = (1.-isq)**2
            n_clu = (eig_vals > l_lbound).sum() # take out all the eigenvalues within and exceeding marcenko-pastur distribution
        if n_clu > 0:
            return pc_trans[:,:n_clu], pc_vecs[:n_clu], eig_vals[:n_clu]
        else:
            print("No correlations founded.")


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
    return CT, V_signif, s

#-------------------------PC clustering method-----------------
def hierachical_pc_clustering(raw_data, n_cut = None, N_iter = 5, mode = -1):
    '''
    cluster the data based on PCA.
    The last coef_pool in the coef_freezer must have one element only.
    n_cut: how many cells does each group have.
    '''
    NT, NC = raw_data.shape
    if n_cut is None:
        n_cut = int(NT/10) # Number of groups to be divided into

    n_slice = _group_division_(NC, n_cut)
    n_groups = n_slice.size # the real n_groups, might increase by 1

    # Initialization 
    coef_freezer = deque()
    TN_series = raw_data

    for nit in range(N_iter):
        '''
        go through iterations
        '''
        print("Iteration:", nit)
        N_effc = 0
        coef_pool = deque()
        ts_pool = []
        gi = 0
        for ng in range(n_groups):
            gf = n_slice[ng]
            data_sub = TN_series[:,gi:gf]
            pc_trans, pc_vecs, eig_vals = pca_dff(data_sub, norm = True, cl_select = mode)
            N_effc += eig_vals.size # number of effective cells
            ts_pool.append(pc_trans)
            coef_pool.append(pc_vecs)
            gi = gf

        # finish one round 
        TN_series = np.column_stack(ts_pool)
        if N_effc <= n_cut:
            print("terminated.")
            break
        else:
            n_slice = _group_division_(N_effc, n_cut)
            n_groups = n_slice.size # new groups
            coef_freezer.append(coef_pool)
        # end for nit

    pc_trans, pc_vecs, eig_vals = pca_dff(TN_series, n_comp = None, norm = True, cl_select = 1)
    coef_freezer.append(pc_vecs)

    # now, finished the whole series. let's do the unpack

    return pc_trans, coef_freezer

def layer_retrieve(coef_int, coef_split):
    '''
    retrieve coefficients of the assembled neurons one layer above.
    coef_int: matrix, SxP
    coef_split: deque with K matrices, each has dimensions p_k*m_k, p_0 + p_1 + .... p_{k-1} = P.
    return: matrix, S*(m_0+m_1+m_2...+m_{k-1}).
    '''
    n_sub = len(coef_split)
    S, P = coef_int.shape
    pm_arr = np.zeros([n_sub, 2])
    m_coef = []
    pi = 0
    for ss in range(n_sub):
        pk, mk = coef_split[ss].shape
        pm_arr[ss, 0], pm_arr[ss, 1] = pk, mk

        coef_subint = coef_int[:,pi:pi+pk]
        m_coef.append(np.matmul(coef_subint,coef_split[ss])) # must use matrix multiplication
        pi = pk # move to the next block

    return np.column_stack(m_coef) # integrated coefficients

def hierachical_pc_unpack(coef_freezer):

    coef_final = coef_freezer.pop() # This must be a matrix
    while coef_freezer:
        cbag = coef_freezer.pop() # pop a bag of coefficients
        m_coef = layer_retrieve(coef_final, cbag)
        coef_final = m_coef

    return coef_final



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
