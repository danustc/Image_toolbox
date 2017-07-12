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


def ic_selector(dff_ica,dff):
    '''
    select ic that have features in the designed time windows.
    '''



# ----------------------Test the ICA function----------------------
def main():
    raw_fname = global_datapath + 'Jun13_A1_GCDA/'
    raw_data = np.load(raw_fname + 'ultra_cleaned.npz')
if __name__ == '__main__':
    main()
