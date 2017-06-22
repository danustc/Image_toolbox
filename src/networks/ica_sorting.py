'''
Independent component analysis(ICA)-based functions
Created by Dan Xie on 05/24/2017
'''
#print(__doc__)
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import matplotlib.pyplot as plt
import src.dynamics.df_f as df_f
import src.visualization.stat_present as stat_present
from sklearn.decomposition import FastICA, PCA
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'



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


def ica_cell_rank(dff_data, i_component = 0):
    '''
    perform ica analysis and evaluate the
    Warning: this function is highly redundant
    '''
    NT, NP = dff_data.shape
    dff_ica, a_mix, s_mean = ica_dff(dff_data, ncomp = NP)
    selected_IC = dff_ica[:,i_component]
    cell_rank = np.argsort(a_mix[:, i_component]**2) # cell contribution to the ICs
    return selected_IC, cell_rank



# ----------------------Test the ICA function----------------------
def main():
    TS_18 = np.load(global_datapath + 'Oct25_B3_TS18.npz')
    TS_data_18 = TS_18['data']

    dff_raw, f_base = df_f.dff_raw(TS_data_18, ft_width=4, ntruncate = 20)

    dff_recon, a_mix = ica_dff(dff_raw[:, a_sorted[:70]],n_comp = 3)
    figi = stat_present.ic_plot(dff_recon)
    figi.savefig(global_datapath+'ica_test_70')

if __name__ == '__main__':
    main()
