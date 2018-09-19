'''
Sample script: clustering of a large group of neurons
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
from src.analysis.Analysis import grinder
import src.algos.spectral_clustering as sc
import matplotlib.pyplot as plt
import clustering

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Jul2017/'
portable_datapath = '/media/sillycat/DanData/'

def main():
    data_path = global_datapath_ubn + 'Jul19_2017_A4_dff.npz'
    N_cut = 1000
    grinder_core = grinder()
    grinder_core.parse_data(data_path)
    grinder_core.activity_sorting()
    grinder_core.background_suppress(sup_coef = 0.0001)
    signal_test = grinder_core.signal[10:, :N_cut]
    peak_position = sc.dataset_evaluation(signal_test)
    print("suggested number of clusters:", peak_position)
    ind_groups, cl_average = clustering.spec_cluster(signal_test, n_clu, threshold = th)


if __name__ == '__main__':
    main()
