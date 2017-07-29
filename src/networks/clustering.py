'''
Created on 07/27/2017 by Dan. Clustering of the data.
Visualization is inherently included here.
Last modification:
'''
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def dis2cluster(dataset, p_levels = None, yield_z = False):


    Z = linkage(dataset, 'ward')
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


def subtree(zmat, N, side = 'L'):
    '''
    recursive searching of subtrees.
    '''
    ind_list = []
    if zmat.size ==4:
        if side =='L':
            return [zmat[0]]
        else:
            return [zmat[1]]
    else:

        if side =='L':
            zind = zmat[-1, 0]
        else:
            zind = zmat[-1, 1] # get the index of subtrees

        if zind < N:
            ind_list +=[zind]
        else:
            ind_list +=subtree(zmat[:zind-N], N, 'L')
            ind_list +=subtree(zmat[:zind-N], N, 'R')

        return ind_list



