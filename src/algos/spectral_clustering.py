'''
My practice of spectral clustering, written from scratch
'''

import numpy as np
import scipy.sparse.linalg as linalg
from scipy.signal import argrelextrema
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
        return L_rw
    else:
        return L


def sc_eigen(L, n_cluster = 20):
    '''
    compute the embedding space of L and cluster. L can be unnormalized or normalized.
    '''
    w, v = linalg.eigsh(L, k = n_cluster, which = 'SA') # compute the 1st n_cluster smallest eigenvalues and eigenvectors.
    print("representing eigenvalues:", w[:n_cluster])

    return w, v



def weakest_connection(corr_mat):
    '''
    check if any individuals are disconnected with anybody else.
    In addition to the weakest link, also compute a general trend of the link strength among the group.
    '''
    nd_mat = corr_mat - np.diag(np.diag(corr_mat))# kill the diagonal elements
    nd_mat[nd_mat < 0] = 0.
    max_corr = nd_mat.max(axis = 0) # the strongest correlation of each cell to everybody else
    sorted_mc = sorted(max_corr) # sort the strongest correlation
    weak_link = sorted_mc[0] # the weakest link



    return weak_link



def corr_afinity(data_raw = None, corr_mat = None, thresh = 0.01, kill_diag = True, adaptive_th = False):
    '''
    Create the afinity matrix
    '''
    if corr_mat is None:
        corr_mat = np.corrcoef(data_raw.T)

    if adaptive_th:
        # adaptive thresholding 
        thresh = weakest_connection(corr_mat)
        print("The real threshold:", thresh)

    corr_mat[corr_mat < thresh] = 1.0e-09

    if kill_diag:
        corr_mat = corr_mat -np.diag(np.diag(corr_mat))
    return corr_mat, thresh


def corr_distribution(corr_mat, nb = 200):
    # set the diagonal to zero first
    corr_mat_sd = corr_mat - np.diag(2*np.diag(corr_mat))
    hist, be = np.histogram(corr_mat_sd.ravel(),  bins = nb, density = True, range = (0,0.4))
    return hist, be



def leigen_nclusters(eigen_list, norder = 0):
    '''
    determining how many clusters the eigenvalue array suggests by finding the kinks in the eigen value list.
    '''
    eig_diff = np.diff(eigen_list)
    peaks = argrelextrema(eig_diff, np.greater)[0] # find kinks

    return peaks[norder]


def dataset_evaluation(raw_data):
    '''
    Have an evaluation of how to set the sc parameters.

    '''
    cmat = np.corrcoef(raw_data.T)
    #weak_link = weakest_connection(cmat)
    aff_mat, th = corr_afinity(data_raw = None, corr_mat = cmat,thresh = 0.01, kill_diag = False, adaptive_th = True ) # first, using adaptive threshold to evaluate the thresholds
    L = laplacian(aff_mat, mode = 'sym') # use the random-walk normalized Laplacian instead of unnormalized version.   
    w, v = sc_eigen(L, n_cluster = 20) # calculate the first 20th eigen values and eigen states
    peak_position = leigen_nclusters(w, norder = np.arange(3)) # where should I cut off?
    return peak_position




