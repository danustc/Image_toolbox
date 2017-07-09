'''
An alternative to PCA sorting. Simple variance of the cell activities.
Added by Dan on 06/22/2017
Last update:
'''
import numpy as np
from collections import deque # my first time trying deque!

def simple_variance_group(dff_raw, n_groups = None):
    '''
    simple variance using merge sort.
    '''

    NT, NP = dff_raw.shape
    if n_group is None:
        n_group = np.max([int(2*NP/NT)+1, 2])

    dff_groups = np.array_split(dff_raw, n_group, axis = 1)
    ind_groups = deque() # create an empty deque
    for dff_sub in dff_groups:
        sub_cov = np.var(dff_sub, axis = 0) # each column is a data set
        ind_groups.append(np.diag(sub_cov))



def simvar_global_sort(dff_raw):
    '''
    load a dff_dataset and calculate the variance, sort the cellular activities by by variance.
    '''
    dvar = -np.var(dff_raw, axis = 0) # calculate the negative variance
    crank = np.argsort(dvar)
    return crank, -dvar



def simvar_layer_clean(dff_raw, var_cut = 0.95, headonly = True):
    '''
    sort a group of data by variance, cut at var_cut
    '''
    crank, gvar = simvar_global_sort(dff_raw)
    sum_var = np.cumsum(gvar)
    sum_var/= sum_var[-1]
    n_cut = np.searchsorted(sum_var, var_cut)
    dff_head = dff_raw[:,crank[:n_cut]]
    dff_tail = dff_raw[:,crank[n_cut:]]
    if headonly:
        return dff_head
    else:
        return dff_head, dff_tail


# --------------------------Below is the test section -------------------------
def main():
    folder_list = glob.glob(global_datapath+'*')
    for folder in folder_list:
        data_merged = np.load(folder+'merged.npz')
        coord_merged = data_merged['coord']
        fluor_merged = data_merged['data']
