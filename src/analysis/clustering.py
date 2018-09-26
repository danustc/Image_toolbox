'''
Created on 07/27/2017 by Dan. Clustering of the data.
Visualization is inherently included here.
Last modification: 09/03/2018
'''
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy import stats
import matplotlib.pyplot as plt
from collections import deque

Z_dic = {'L':0, 'R':1}


def dis2cluster(dataset, p_levels = None, yield_z = False):
    '''
    dendrogram clustering
    '''
    Z = linkage(dataset, 'ward') # this should be paid more attention to.
    c, coph_dists = cophenet(Z,pdist(dataset))
    print(c)
    figc = plt.figure()
    ax = figc.add_subplot(111)
    ax.set_ylabel('distance')
    if p_levels is None:
        R = dendrogram(Z, leaf_rotation = 90.)
    else:
        R = dendrogram(Z, leaf_rotation = 90.,
            show_contracted = True,
            truncate_mode = 'level',
            p=p_levels)
    if yield_z:
        return figc, R, Z
    else:
        return figc, R


def subtree(zmat, N, side = 'L', root_node = False):
    '''
    recursive searching of subtrees from a linkage matrix
    OK this works!
    '''
    if zmat.size ==4:
        return [int(zmat[0, Z_dic[side]])] # if zmat is 1-D array, then there is no need to worry about the root node. 
    else:
        nm = zmat[-1,3] # the multiplicity
        if nm ==2:
            return [int(zmat[-1, Z_dic[side]])] # if the multiplicity is 2, i.e., the two leaves are primary leaves, there is also no need to worry about the root node.
        else:
            ind_list = []
            zind = int(zmat[-1, Z_dic[side]])

            if zind < N:
                return [zind] # again, if the leave is primary, no need to worry about the root node. 
            else:
                ind_list +=subtree(zmat[:zind-N+1], N, 'L')
                ind_list +=subtree(zmat[:zind-N+1], N, 'R')
        if root_node:
            ind_list.append(zind-N+1) # find the root node of the subtree. 
        return ind_list
    # -------------------end of subtree


def assert_subtree(dmat, ind_list):
    '''
    check if a subtree can be separated cleanly from a big tree.
    OK this also works. Ind_list cannot have the root node index.
    '''
    d_subtree = dmat[ind_list]
    z_subtree = linkage(d_subtree, 'ward')
    fig_sbt = plt.figure()
    ax = fig_sbt.add_subplot(111)
    ax.set_ylabel('distance')
    R = dendrogram(z_subtree,leaf_rotation = 90.)
    return fig_sbt, R


def histo_clustering(feature, nbin, bin_cut = None,n_fold = 2, sca = 1.00):
    '''
    feature: the histogram of features
    bin_range: the range of the bins
    n_fold: the fold factor of bins
    '''
    hist, bins = np.histogram(feature, bins = nbin)
    xbin = bins[1] - bins[0]
    #    plt.hist(feature, bins = nbin)

    if bin_cut is not None:
        norm_feature = feature[feature<bin_cut]
        res_feature = feature[feature>=bin_cut]
        m, s = stats.norm.fit(norm_feature) # m: mean, #s: spreading
        pdf_g = stats.norm.pdf(bins, m, s)*len(norm_feature)*xbin # spread functin
    else:
        m, s = stats.norm.fit(feature) # no cutting off
        pdf_g = stats.norm.pdf(bins, m, s)*len(feature)*xbin
        # spread functin

    mpdf = (pdf_g[:-1]+pdf_g[1:])//2
    res_hist = hist - np.floor(mpdf*sca)
    res_hist[res_hist<0] = 0
    n_padding = nbin%n_fold
    if n_padding:
        res_hist = np.append(res_hist, np.zeros(n_fold -n_padding))
    # next, merge the bins
    n_rows = len(res_hist) //n_fold
    merged_hist = np.sum(np.reshape(res_hist, (n_rows, n_fold)),axis = 1)
    mb_centers = np.arange(n_rows)*xbin*n_fold+(bins[0]+bins[1])*0.50

    # to be added: find the cut off.

    return merged_hist, mb_centers

