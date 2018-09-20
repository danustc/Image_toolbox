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

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Jul2017/'
portable_datapath = '/media/sillycat/DanData/'

def main():
    data_path = global_datapath_ubn + 'Jul19_2017_A4_dff_cleaned.npz'
    dff_data = np.load(data_path)
    signal_test = dff_data['signal']
    coord_test = dff_data['coord']
    igp, cap = sc.hierachical_sc(signal_test, n_group = 8, threshold = 0.25, interactive = True)



if __name__ == '__main__':
    main()
