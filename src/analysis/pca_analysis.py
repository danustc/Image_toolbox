'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
sys.path.append('/home/sillycat/Programming/Python/tsne_python/')
import tsne
from Analysis import grinder, coregen
from collections import deque
from pca_funcs import *
from src.visualization.stat_present import direct_dimplot
import matplotlib.pyplot as plt
global_datapath = '/home/sillycat/Programming/Python/data_test/'


def hierachical_pc_clustering_label(raw_data, glabels, N_iter = 5, mode = -1):
    '''
    pca clustering method for predetermined groups of data.
    '''
    NT, NC = raw_data.shape
    unique_labels = np.unique(glabels)
    group_list = []
    for label in unique_labels:
        ind_group = np.where(glabels == label)[0]
        group_list.append(ind_group)


class analysis(object):
    '''
    Perform PCA on a group of data with more columns on rows
    Divide groups: direct divide, label divide
    '''
    def __init__(self, grinder_core = None):
        '''
        Load data. Specify the variance cut-off.
        '''
        if grinder_core is None:
            grinder_core = grinder()
        self._grinder = grinder_core

    def hierachical_pc_clustering_label(self):
        pass
#----------------------------------------------Test the function-------------------------------------------

if __name__ == '__main__':
    '''
    The main function is imported from Analysis
    '''
    grinder_core = coregen()
    glabel = grinder_core.spatial_gridding((3,4,5))
    print(glabel)
    Anna = analysis(grinder_core)
    pc_trans, pc_vecs, eigs = pca_dff(Anna._grinder.signal[5:, :400], cl_select = 1)
    fig_p = direct_dimplot(pc_vecs[:7].T)
    fig_p.savefig(global_datapath+'test_pca_d')
     
