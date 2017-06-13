'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA, FastICA
import src.dynamics.df_f as df_f # the functions of calculating dff
import src.visualization.stat_present as stat_present
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/HQ/'


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



def pca_group(data, n_group= 5, var_cut = 0.90):
    '''
    split the raw data set into several groups, and do PCA analysis for each group.
    '''
    N, P = data.shape
    data_group = np.array_split(data, n_group, axis = 1) # split the raw data into several sections.

    for sgroup in data_group:
        # do PCA analysis on each subgroup and merge all of them
        pass

    return CT, V


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
    ncut = int(NP/2)+1
    ind_front = arg_sv[:ncut] # split arg_sv into two groups
    int_back = arg_sv[ncut:]

    dff_front = dff_data[:, ind_front]
    dff_back = dff_data[:, ind_back]
    cov_front = np.cov(dff_front, rowvar =False) # calculate the covariances
    cov_back = np.cov(dff_back, rowvar = False)








#----------------------------------------------Test the function-------------------------------------------

def main():
    '''
    Test written on 04/30/2017.
    '''
    #TS_9 = np.load(global_datapath+'TS_9.npz')
    Dec07_B1 = np.load(global_datapath + 'Dec06_2016_A1_merge_fluo.npz')
    Dec07_B1_data = Dec07_B1['merge_003']

    dff_raw, f_base = df_f.dff_raw(Dec07_B1_data, ft_width=4, ntruncate = 20)
    CT, V = pca_raw(dff_raw, var_cut = 0.95)
    print(V.shape)
    fig = stat_present.PCA_trajectory_matrix(CT, dim_select = [0,1,2,3,4])
    fig.savefig(global_datapath+'pc_test_habe')
    figv = stat_present.pc_component_grid(V[:,:100],npc = 55)
    figv.savefig(global_datapath+'pc_celldistri')
    a_sorted = cell_sorting(V)

    nselect = 20
    asv = a_sorted[:nselect]
    isv = a_sorted[-nselect:]
    dff_select = dff_raw[:,asv]
    figd = stat_present.nature_style_dffplot(dff_select, dt = 0.5, sc_bar = 0.50)
    figd.savefig(global_datapath+'most_active_100_10')
    dff_select = dff_raw[:,isv]
    figd = stat_present.nature_style_dffplot(dff_select, dt = 0.5, sc_bar = 0.10)
    figd.savefig(global_datapath+'least_active_100_10')
    figr = stat_present.dff_rasterplot(dff_raw[:,a_sorted[:100]])
    figr.savefig(global_datapath+'raster_test', tunit = 'm')
    #independent component analysis
    dff_recon, a_mix = ica_dff(dff_raw[:, a_sorted[:70]],n_comp = 3)
    figi = stat_present.ic_plot(dff_recon)
    figi.savefig(global_datapath+'ica_test_70')

if __name__ == '__main__':
    main()


