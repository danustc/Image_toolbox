'''
Sample script: clustering of a large group of neurons
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
from src.single_analysis.Analysis import grinder
import src.analysis.spectral_clustering as sc
import matplotlib.pyplot as plt
from hierachical_sc import hrc_sc
from src.visualization.signal_plot import compact_dffplot

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Jul2017/'
portable_datapath = '/media/sillycat/DanData/'

def main():
    data_path = global_datapath_ubn + 'Jul19_2017_A4_dff_cleaned.npz'
    dff_data = np.load(data_path)
    signal_test = dff_data['signal'][10:]
    coord_test = dff_data['coord']
    HS_class = hrc_sc(signal_test, n_group = 10)
    print(HS_class.__dict__.keys())
    HS_class.divide_sc(threshold = 0.25)
    HS_class.groupwise_population_labeling()
    cluster_cg = HS_class.cluster_corrcheck()
    merged_label, cl_average, cind = HS_class.merge_clusters(cluster_cg)


    indall = np.concatenate(cind)
    print(indall.size)
    print(len(set(indall)))
    fig_merged = compact_dffplot(cl_average, fsize = (6,3.0))
    fig_merged.savefig('sc_merged')
    #print(HS_class.__dict__.keys())




if __name__ == '__main__':
    main()
