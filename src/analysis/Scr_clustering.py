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
from src.visualization.cluster_navigation import clusters_show
from spatial import coord_cluster

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Aug2018/homo/'
portable_datapath = '/media/sillycat/DanData/'




def view_clusters(BG_image):
    pass


def main():
    data_path = global_datapath_ubn + 'Aug23_2018_B4_ref_lb_cleaned.npz'
    dff_data = np.load(data_path)
    signal_test = dff_data['signal'][10:,:5000]
    coord_test = dff_data['coord'][:5000]
    HS_class = hrc_sc(signal_test, n_group = 8)
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
        fig_raster.savefig('cluster_'+str(ii))
        plt.close(fig_raster)


    cc = coord_cluster(coord_test, cind)
    fig_coord = clusters_show(cc[:5])
    fig_coord.savefig('clshow')
    fig_merged = compact_dffplot(cl_average, fsize = (6,4))
    fig_merged.savefig('sc_merged')




if __name__ == '__main__':
    main()
