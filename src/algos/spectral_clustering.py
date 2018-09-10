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
    D_sum = W.sum(axis = 1) # diagonal matrix
    D = np.diag(D_sum)
    print(D.shape)
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
    w, v = linalg.eigsh(L, k = n_cluster, which = 'SA') # compute the 1st n_cluster smallest eigenvalues and eigenvectors.
    print("representing eigenvalues:", w[:n_cluster])

    return w, v


def corr_afinity(data_raw, thresh = 0.01, kill_diag = True):
    '''
    Create the afinity matrix
    '''
    corr_mat = np.corrcoef(data_raw.T)
    corr_mat[corr_mat < thresh] = 1.0e-09
    if kill_diag:
        corr_mat = corr_mat -np.diag(np.diag(corr_mat))
    return corr_mat


def n_clusters(eigen_list):
    '''
    determining how many clusters the eigenvalue array suggests
    '''
    eig_diff = np.diff(eigen_list)
