'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA
import src.dynamics.df_f as df_f # the functions of calculating dff
from src.visualization.stat_present import PCA_scatter_matrix, PCA_trajectory_matrix, pc_component_grid
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
def cell_sorting(V, nselect = 10):
    '''
    classifying the cells into active and non-active groups according to their V coefficients. V contains the principal vectors sorted by the eigen values.
    returns the most and least active nselect neurons.
    '''
    NV, NP = V.shape
    sv = np.sum(-V**2, axis = 0)
    arg_sv = np.argsort(sv)
    print(sv[arg_sv])
    return arg_sv[:nselect], arg_sv[-nselect:]





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


#----------------------------------------------Test the function-------------------------------------------

def main():
    '''
    Test written on 04/30/2017.
    '''
    #TS_9 = np.load(global_datapath+'TS_9.npz')
    TS_18 = np.load(global_datapath + 'Oct25_B3_TS18.npz')
    TS_data_18 = TS_18['data']


    dff_raw, f_base = df_f.dff_raw(TS_data_18[:,:100], ft_width=4, ntruncate = 40)
    print(dff_raw.shape)
    CT, V = pca_raw(dff_raw, var_cut = 0.95)
    print(CT.shape)
    print(V.shape)
    fig = PCA_trajectory_matrix(CT, dim_select = [0,1,2,3,4])
    fig.savefig(global_datapath+'pc_test_habe')
    figv = pc_component_grid(V,npc = 37)
    figv.savefig(global_datapath+'pc_celldistri')
    asv, isv = cell_sorting(V, nselect=50)
    dff_select = dff_raw[:,asv]
    figd = df_f.nature_style_dffplot(dff_select, dt = 0.5, sc_bar = 0.50)
    figd.savefig(global_datapath+'most_active_100_10')
    dff_select = dff_raw[:,isv]
    figd = df_f.nature_style_dffplot(dff_select, dt = 0.5, sc_bar = 0.10)
    figd.savefig(global_datapath+'least_active_100_10')



if __name__ == '__main__':
    main()


