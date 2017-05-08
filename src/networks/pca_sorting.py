'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA
import src.dynamics.df_f as df_f # the functions of calculating dff
from src.visualization.stat_present import PCA_scatter_matrix, PCA_trajectory_matrix
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
def pca_raw(data, n_comp = None, var_cut = 0.95):
    '''
    data: uncentralized, unstandardlized data
    n_comp: the number of components to be extracted.
    var_cut: variance cut, ignore the contributions from minor principal components.
    '''
    N, P = data.shape # the number of measurements and the number of 'sensors'
    c_data = data-data.mean(axis = 0) # centralization
    U,s,V = np.linalg.svd(c_data) # svd 
    lam = s**2/(N-1)
    if n_comp is None:
        var_tot = lam.cumsum()/lam.sum() # 
        n_comp = np.searchsorted(var_tot, var_cut)+1 # the number of principal components that covers var_cut fraction of variance
    CT = U[:,:n_comp]*s[:n_comp] # the coefficients on the chosen PCs  
    return CT, V


# Next, try to remove the noisy cells which are not firin



def pca_group(data, n_group= 5, var_cut = 0.95):
    '''
    split the raw data set into several groups, and do PCA analysis for each group.
    '''
    N, P = data.shape
    data_group = np.array_split(data, n_group, axis = 1) # split the raw data into several sections.



    return CT, V

#----------------------------------------------Test the function-------------------------------------------

def main():
    '''
    Test written on 04/30/2017.
    '''
    TS_9 = np.load(global_datapath+'TS_9.npz')
    TS_14 = np.load(global_datapath+'TS_14.npz')
    TS_data_09 = TS_9['data']
    TS_data_14 = TS_14['data']

    TS_data = np.hstack((TS_data_09, TS_data_14))

    dff_raw, f_base = df_f.dff_raw(TS_data_09[:,:100], ft_width=4, ntruncate = 50)
    print(dff_raw.shape)
    CT, V = pca_raw(dff_raw, var_cut = 0.95)
    print(CT.shape)
    fig = PCA_trajectory_matrix(CT, dim_select = [0,1,2,3,4])
    fig.show()
    fig.savefig(global_datapath+'pc_test')

if __name__ == '__main__':
    main()


