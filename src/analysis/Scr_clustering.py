'''
Sample script: clustering of a large group of neurons
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
from src.analysis.Analysis import grinder
import src.analysis.spectral_clustering as sc
import matplotlib.pyplot as plt
from hierachical_sc import hrc_sc

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Jul2017/'
portable_datapath = '/media/sillycat/DanData/'

def main():
    data_path = global_datapath_ubn + 'Jul19_2017_A4_dff_cleaned.npz'
    dff_data = np.load(data_path)
    signal_test = dff_data['signal']
    coord_test = dff_data['coord']
    HS_class = hrc_sc(signal_test, n_group = 5)
    print(HS_class.__dict__.keys())
    HS_class.divide_sc(threshold = 0.25)
    HS_class.population_labeling()
    HS_class.cluster_corrcheck()
    print(HS_class.__dict__.keys())




if __name__ == '__main__':
    main()
