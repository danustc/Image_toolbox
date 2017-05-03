'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
from sklearn.decomposition import PCA
import src.dynamics.df_f as df_f # the functions of calculating dff
from src.visualization.stat_present import PCA_scatter_matrix
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

    dff_raw, f_base = df_f.dff_raw(TS_data, ft_width=4, ntruncate = 50)
    pc_trans = pca_dff(dff_raw, n_comp = 5)
    fig = PCA_scatter_matrix(pc_trans, dim_select = [0,1,2])
    fig.show()
    fig.savefig(global_datapath+'pc_test')

if __name__ == '__main__':
    main()


