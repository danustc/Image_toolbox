'''
Independent component analysis(ICA)-based functions
Created by Dan Xie on 05/24/2017
'''
#print(__doc__)
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import matplotlib.pyplot as plt
import src.visualization.stat_present as stat_present
import src.networks.clustering as clustering
from sklearn.decomposition import FastICA
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/HQ/'



def ica_dff(dff_data, n_comp = 4, stdize = False):
    '''
    directly use the ICA algorithm in sklearn
    '''
    if stdize:
        dff_input =dff_data/dff_data.std(axis = 0)# standardize data
    else:
        dff_input = dff_data
    ica = FastICA(n_components = n_comp)
    dff_ica = ica.fit_transform(dff_input) # reconstruct signals
    a_mix = ica.mixing_ # the estimated mixing matrix. Row: # of cells; column: # of ICs.
    s_mean = ica.mean_
    # the original data can be recovered by np.dot(dff_ica, a_mix.T) + s_mean. 
    return dff_ica , a_mix, s_mean



# ----------------------Test the ICA function----------------------
def main():
    n_ica = 3
    n_select = 100
    raw_fname = global_datapath + 'Nov01_2016_A1/'
    dff_data = np.load(raw_fname + 'merged_dff.npz')
    dff_signal = dff_data['signal']
    dff_ica, a_mix, s_mean = ica_dff(dff_signal[:,:n_select], n_comp = n_ica)

    figc, R, Z = clustering.dis2cluster(a_mix, yield_z = True)
    figc.savefig(raw_fname+'/ic_cluster_'+str(n_ica))
    pid = np.array(R['icoord'])
    pdd = np.array(R['dcoord'])
    pco = np.array(R['color_list'])
    leaves = R['leaves']
    figi = stat_present.ic_plot(dff_ica, dt = 0.5)
    figi.savefig(raw_fname+'/ic_'+str(n_ica))
    ind_list_L = clustering.subtree(Z, n_select, 'L', False)
    ind_list_R = clustering.subtree(Z, n_select, 'R', False)
    print(ind_list_L)
    print(ind_list_R)
    cluster_indices = [ind_list_L, ind_list_R]
    figl = stat_present.cluster_dimplot(a_mix, cluster_indices, 'rg')
    figl.savefig(raw_fname + '/ic_cluster_scatter')

    figs, R = clustering.assert_subtree(a_mix, ind_list_R)
    figs.savefig(raw_fname + '/ic_subtree')

if __name__ == '__main__':
    main()
