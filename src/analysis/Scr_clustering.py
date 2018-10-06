'''
Sample script: clustering of a large group of neurons
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
from src.analysis.single_analysis import grinder
import src.analysis.spectral_clustering as sc
import matplotlib.pyplot as plt
from hierachical_sc import hrc_sc
from src.visualization.signal_plot import compact_dffplot, dff_rasterplot
from src.visualization.cluster_navigation import multi_cluster_show
import tifffile as tf
from spatial import coord_cluster

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Aug2018/'
portable_datapath = '/media/sillycat/DanData/'




def main():
    fish_name  = 'Aug23B4'
    data_path = global_datapath_ubn + 'Aug23_2018_B4_ref_cleaned.npz'
    vol_img = tf.imread(global_datapath_ubn+'MAX_Aug23_B4_ref.tif')
    dff_data = np.load(data_path)
    signal_test = dff_data['signal'][5:,:5500]
#    ccc = sc.Corr_sc()
#    ccc.load_data(signal_test)
#    ccc.link_evaluate(sca = 1.12)
#    ccc.affinity()
#    ccc.laplacian_evaluation()
#
#
#    ccc.clustering(n_clu = 14)
#    fig_ccc = compact_dffplot(ccc.cl_average, fsize = (6,6))
#    fig_ccc.savefig('direct_B4')
#    plt.close(fig_ccc)
#
#    direct_pop = [len(ig) for ig in ccc.ind_groups]
#    for ii in range(14):
#        ci = ccc.ind_groups[ii]
#        cluster_signal = signal_test[:, ci]
#        fig_raster = dff_rasterplot(cluster_signal)
#        fig_raster.savefig(fish_name + 'straight_'+str(ii))
#        plt.close(fig_raster)
#
#    coord_test = dff_data['coord'][:5500]
#    cc = coord_cluster(coord_test, ccc.ind_groups)
#
#    fig_coord = multi_cluster_show(cc[:12], vol_img, cm = 'viridis', layout = (4,3), sup_title = fish_name, fsize = (7.5, 8.2))
#    fig_coord.savefig('clshow_' + fish_name)
#
    HS_class = hrc_sc(signal_test, n_group = 15)
    print(HS_class.__dict__.keys())
    HS_class.divide_sc(mode = 'random')
    HS_class.groupwise_population_labeling()
    cluster_cg = HS_class.cluster_corrcheck()
    merged_label, cl_average, cind = HS_class.merge_clusters(cluster_cg)
    NT, NC = cl_average.shape

    for ii in range(NC):
        ci = cind[ii]
        cluster_signal = signal_test[:, ci]
        fig_raster = dff_rasterplot(cluster_signal)
        fig_raster.savefig('dq_'+str(ii))
        plt.close(fig_raster)

    dq_pop = [len(ig) for ig in cind]

    fig_merged = compact_dffplot(cl_average, fsize = (6,6))
    fig_merged.savefig('dq_B4')

    plt.close(fig_merged)



if __name__ == '__main__':
    main()
