'''
Created on 07/27/2017 by Dan. Clustering of the data.
Visualization is inherently included here.
Last modification: 11/05/2017
'''
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
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


def histo_clustering(feature_hist, bin_range, n_fold = 2, ridc = 0.):
    '''
    feature_hist: the histogram of features
    bin_range: the range of the bins
    n_fold: the fold factor of bins
    ridc: the gaussian distribution of the background center
    '''

