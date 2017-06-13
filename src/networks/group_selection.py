'''
Last modification: 04/28/2017 by Dan
This module contains network analysis. ... to be filled up.

'''
import glob
import numpy as np
import random
from pca_sorting import *


def group_pca_sorting(raw_signals, dim_red = 0.50, Niter = 3, var_cut = 0.95):
    '''
    raw_signals: a large matrix with # of columns larger than # of rows.
    dim_red: each group has # of columns as a fraction of # of rows.
    Niter: the iterations of shuffling and re-sorting
    var_cut: the fraction of overall variance
    '''
    NT, NP = raw_signals.shape
    ind_mark = np.arange(NP) # the index marker array

    for ni in range(Niter):
        pass

    return group

