'''
Independent component analysis(ICA)-based functions
Created by Dan Xie on 05/24/2017
Last update: 11/08/2017
'''
#print(__doc__)
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import matplotlib.pyplot as plt
import src.visualization.stat_present as stat_present
import src.analysis.clustering as clustering
from sklearn.decomposition import FastICA
from sklearn import linear_model
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'



def ica_dff(dff_data, n_comp = 4, stdize = False, posi_restrict = True):
    '''
    directly use the ICA algorithm in sklearn
    a_mix: NCxNI matrix, NI: # of independent components
    '''
    print("ICA.")
    if stdize:
        dff_input =dff_data/dff_data.std(axis = 0)# standardize data
    else:
        dff_input = dff_data
    ica = FastICA(n_components = n_comp)
    dff_ica = ica.fit_transform(dff_input) # reconstruct signals
    a_mix = ica.mixing_ # the estimated mixing matrix. Row: # of cells; column: # of ICs.
    s_mean = ica.mean_
    if posi_restrict:
        di_max = np.max(dff_ica, axis = 0)
        di_min = np.min(dff_ica, axis = 0)
        diff_loc = di_max + di_min
        flip_idx =  np.where(diff_loc<0)[0]
        if flip_idx.size >0:
            dff_ica[:, flip_idx]*=-1
            a_mix[:, flip_idx]*=-1

    # the original data can be recovered by np.dot(dff_ica, a_mix.T) + s_mean. 
    return dff_ica , a_mix, s_mean

def ica_regression(ics, new_data):
    '''
    Test ica_regression
    '''
    lr = linear_model.LinearRegression()
    lr.fit(ics, new_data)
    a_mix = lr.coef_
    return a_mix

# ----------------------Test the ICA function----------------------
def main():
    n_ica = 3
    n_select = 200
    raw_fname = global_datapath + 'Jun13_A1_GCDA/'
    dff_data = np.load(raw_fname + 'merged_dff.npz')
    dff_signal = dff_data['signal']
    print(dff_signal.shape)
    dff_ica, a_mix, s_mean = ica_dff(dff_signal[:,:n_select], n_comp = n_ica)

    figi = stat_present.ic_plot(dff_ica, dt = 0.5)
    figi.savefig(raw_fname+'/ic_'+str(n_ica))

if __name__ == '__main__':
    main()
