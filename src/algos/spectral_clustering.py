'''
My practice of spectral clustering, written from scratch
'''

import numpy as np
import scipy.sparse.linalg as linalg
from sklearn.cluster import KMeans


def laplacian(W, mode = 'un'):
    '''
    construct graph laplacian from the weight matrix
    there are three modes:
        un: unnormalized
        sym: symmetrically normalized
        rw: random walk-type, normalized
    '''
    D_sum = np.diag(W.sum(axis = 0)) # diagonal matrix
    D = np.diag(D_sum)
    L = D - W
    D_isq = np.diag(1./(D_sum**0.5))
    if mode =='sym':
        L_sym = np.dot(D_isq, L).dot(D_isq)
        return L_sym
    elif mode == 'rw':
        L_rw = np.diag(1./D_sum).dot(L)
    else:
        return L


def sc_unnormalized(L, n_cluster = 8):
    '''
    compute the embedding space of L and cluster.
    '''
    NL = L.shape[0] # the number of NL 
    w, v = linalg.eigsh(L, k = n_cluster, which = 'SM') # compute the 1st n_cluster smallest eigenvalues and eigenvectors.
    print("representing eigenvalues:", v)


