'''
spatial analysis of clusters
Created by Dan on 09/25/2018
'''
import numpy as np

from scipy.ndimage.filters import gaussian_filter


def coord_cluster(coords, cind_label):
    '''
    group coordinates into clusters based on labeling.
    '''
    NL = len(cind_label)
    cc = []
    for ci in cind_label:
        cc.append(coords[ci])

    return cc


def mutual_information_cloud(cloud_a, cloud_b):
    '''
    calculate mutual information between two clouds.
    Needs to design this algorithm carefully.
    '''



def dist_pointcloud(coords, x_range, y_range, dbin = 0.30, smooth_sig = 1.0 ):
    '''
    coords: 2-column array, ordered in x-y.
    nr: bin width
    '''
    # evaluate the distribution 
    xbin = int(x_range//dbin)
    ybin = int(y_range//dbin)
    sig = smooth_sig/dbin # the sigma in the unit of dbin
    sig_2d = [sig, sig]
    H, y_edges, x_edges = np.histogram2d(coords[:,1], coords[:,0], bins = (ybin, xbin), density = False) # in H, x-axis is the rows and y-axis is the columns
    SH = gaussian_filter(H, sig_2d, mode = 'constant') # smooth.
    norm_fact = SH.sum()*dbin*dbin # normalize with the integral
    SH = SH/norm_fact

    return SH




