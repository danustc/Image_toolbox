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
        sub_cov = np.cov(dff_sub,rowvar = False) # each column is a data set
        ind_groups.append(np.diag(sub_cov))
