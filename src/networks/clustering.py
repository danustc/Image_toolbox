'''
Created on 07/27/2017 by Dan. Clustering of the data.
Visualization is inherently included here.
Last modification:
'''
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

Z_dic = {'L':0, 'R':1}

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
    recursive searching of subtrees from a linkage matrix
    OK this works!
    '''
    print(side)
    if zmat.size ==4:
        return [int(zmat[0, Z_dic[side]])]
    else:
        nm = zmat[-1,3] # the multiplicity
        if nm ==2:
            return [int(zmat[-1, Z_dic[side]])]
        else:
            ind_list = []
            zind = int(zmat[-1, Z_dic[side]])

            if zind < N:
                return [zind]
            else:
                ind_list +=subtree(zmat[:zind-N+1], N, 'L')
                ind_list +=subtree(zmat[:zind-N+1], N, 'R')

            return ind_list
    # -------------------end of subtree



