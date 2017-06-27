'''
Redundancy detection between different slices
'''
import os
import pandas as pd
import numpy as np


def redund_detect_merge(coord_ref, coord_fol, thresh = 2.0):
    '''
    Compare the redundancy and merge two slices
    coord_ref, coord_fol: the two matrices of y-x coordinates in the reference slice and the following slice
    Comments on 04/13: this is not well-designed, but let's see if it works.
    '''
    y1 = coord_ref[:,0]
    x1 = coord_ref[:,1]
    y2 = coord_fol[:,0]
    x2 = coord_fol[:,1]

    n1 = len(y1) # the length of the arrays
    n2 = len(x1)



    [YC, YR] = np.meshgrid(y2,y1)
    [XC, XR] = np.meshgrid(x2,x1)
    dist_block = np.sqrt((YC-YR)**2 + (XC-XR)**2)
    red_pair = np.where(dist_block <= thresh)

    ind1 = red_pair[0]
    ind2 = red_pair[1]

    return ind1, ind2 # only return indices




