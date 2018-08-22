'''
Cell segmentation algorithm based on the Cytometry paper.
Created by Dan on 08/09/2018.
'''
import numpy as np


def hist_group(img, nbins = 200, n_group = 2):
    '''
    img: raw image
    n_group: number of groups to be divided into
    '''

    hist, be = np.histogram(img, nbins)

