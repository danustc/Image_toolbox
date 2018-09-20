'''
My practice of spectral clustering, written from scratch
'''

import numpy as np
import scipy.sparse.linalg as linalg
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from collections import deque


def symmetric(mat, tol = 1.0e-08):
    return np.allclose(mat, mat.T, atol = tol)

def smart_partition(NC, n_group, last_big = True):
    arr = np.arange(NC)
    if last_big:
        g_pop = int(NC//n_group)
    else:
        g_pop = int(NC//n_group) + 1
    cutoff_pos = np.arange(n_group+1) * g_pop
    cutoff_pos[-1] = NC #reset the last element to NC
    group_index = []
    for ii in range(n_group):
        ni = cutoff_pos[ii]
        nf = cutoff_pos[ii+1]
        group_index.append(arr[ni:nf])

    return group_index


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
        # warning: the eigen value of L_rw is not all non-negative. I need to double check the tutorial.
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
    diff_smc = np.diff(sorted_mc) # differential link strength
    peak = argrelextrema(diff_smc, np.greater)[0] # find peaks in the sorted sorted mc.
    if len(peak) ==0:
        weak_link = sorted_mc[0]
    else:
        weak_link = sorted_mc[:peak[0]+1] # give a list of cut-off options

    return weak_link, peak



def corr_afinity(data_raw = None, corr_mat = None, thresh = 0.01, kill_diag = True, adaptive_th = False):
    '''
    Create the afinity matrix
    The cutoff is radical
    '''
    if corr_mat is None:
        corr_mat = np.corrcoef(data_raw.T)

    if adaptive_th:
        # adaptive thresholding 
        weak_link, peak = weakest_connection(corr_mat)
        if len(peak):
            thresh = weak_link[0]
        else:
            thresh = weak_link
        print("The real threshold:", thresh)

    corr_mat[corr_mat < thresh] = 1.0e-09
    if symmetric(corr_mat):
        pass
    else:
        print("Affinity matrix not symmetric.")
        print((corr_mat-corr_mat.T).max())
        corr_mat = (corr_mat + corr_mat.T)*0.5

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
    peaks = argrelextrema(eig_diff, np.greater)[0]+1 # find kinks

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
    return peak_position, th


def spec_cluster(raw_data, n_cl = 5, threshold = 0.05, average_calc = True):
    '''
    raw_data: NT x NC, NT: # of trials, NC: # of cells
    perform spectral clustering
    threshold: correlation coefficient lower than the threshold is considered unconnected.
    Add: calculate cluster population and average, then order by population size.
    '''
    # Create a distant matrix
    NT, NC = raw_data.shape
    affi_mat, th = corr_afinity(raw_data, thresh = threshold )
    SC = SpectralClustering(n_clusters = n_cl, affinity = 'precomputed')
    y_labels = SC.fit_predict(affi_mat)
    total_ind = np.arange(NC)
    ind_groups = []
    g_population = np.zeros(n_cl)

    for ii in range(n_cl):
        ind_clu = total_ind[ y_labels == ii]
        ind_groups.append(ind_clu)
        g_population[ii] = len(ind_clu)

    sort_pop = np.argsort(g_population).astype('int')
    ind_groups = [ind_groups[sort_pop[ii]] for ii in range(n_cl)]
    if average_calc:
        cl_average = np.zeros([NT, n_cl])
        for ii in range(n_cl):
            cl_average[:,ii] = raw_data[:, ind_groups[ii]].mean(axis = 1)

    else:
        cl_average = None
    return ind_groups,  cl_average


def hierachical_sc(raw_data, n_group, threshold = 0.25, mode = 'random', interactive = True):
    '''
    spectral clustering by layers.
    hard part: how to keep tracing the clustering results.
    This can be full automatic or semi-automatic.
    '''
    NT, NC = raw_data.shape
    arr = np.arange(NC)
    group_index = smart_partition(NC, n_group, last_big = False) # Equally partition the dataset into several groups, the last group has the smallest population.

    if mode == 'random':
        np.random.shuffle(arr)

    elif mode == 'ordered':
        pass

    cl_average_pool = deque() # list of lists, saving the cluster average
    ind_group_pool = deque() # list of lists, saving the group index average
    for gg in range(n_group): # iterate over n_group
        '''
        first, evaluate the group's threshold
        '''
        sg_data = raw_data[:,group_index[gg]] # takeout a subgroup of data
        cluster_peaks, th = dataset_evaluation(sg_data)
        print("suggested number of clusters:", cluster_peaks)
        print("suggested threshold:", th)

        if interactive:
            n_cl = int(input("Enter the number of clusters: "))
        else:
            if len(cluster_peaks) == 1:
                n_cl = peak_position[0]
            else:
                n_cl = peak_position[1]
        ind_groups, cl_average = spec_cluster(sg_data, n_cl)
        ind_group_pool.append(ind_groups)
        cl_average_pool.append(cl_average)

    return ind_group_pool, cl_average_pool





