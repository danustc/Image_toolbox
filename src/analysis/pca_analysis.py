'''
This is a trial module for PCA analysis of large-scale calcium imaging data.
Based on scikit-learn
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import Analysis
from collections import deque
from pca_funcs import *
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
        pass


class analysis(object):
    '''
    Perform PCA on a group of data with more columns on rows
    Divide groups: direct divide, label divide
    '''
    def __init__(self, coord = None, signal = None):
        '''
        Load data. Specify the variance cut-off.
        '''
        print("The class is initiated.")
        if signal is not None:
            print(signal.shape)

#----------------------------------------------Test the function-------------------------------------------

def main():
    signal = np.random.randn(1785, 10)
    a = analysis(coord = None, signal = signal)
if __name__ == '__main__':
    main()


