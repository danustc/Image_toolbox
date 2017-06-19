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



def ica_dff(dff_data, n_comp = 4):
    '''
    directly use the ICA algorithm in sklearn
    '''
    dff_std = dff_data/dff_data.std(axis = 0)# standardize data
    ica = FastICA(n_components = n_comp)
    dff_recon = ica.fit_transform(dff_std) # reconstruct signals
    a_mix = ica.mixing_ # the estimated mixing matrix
    return dff_recon, a_mix




# ----------------------Test the ICA function----------------------
def main():
    TS_18 = np.load(global_datapath + 'Oct25_B3_TS18.npz')
    TS_data_18 = TS_18['data']

    dff_raw, f_base = df_f.dff_raw(TS_data_18, ft_width=4, ntruncate = 20)


if __name__ == '__main__':
    main()
