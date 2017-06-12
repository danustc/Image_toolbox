'''
Last modification: 04/28/2017 by Dan
This module contains network analysis. ... to be filled up.

'''

import numpy as np

def spatial_cell_selection(coord_data, coordinates, tolerance = 5):
    '''
    data: contains the coordinates, column0: y, column1:x
    coordinates: the centers of selection
    tolerance: allowed error from the center, in pixels
    '''
    yc = coord_data[:,0]
    xc = coord_data[:,1]

    [YS, YC] = np.meshgrid(coordinates[:,0], yc)
    [XS, XC] = np.meshgrid(coordinates[:,1], xc)

    r_diff = np.sqrt((YC-YS)**2 + (XC-XS)**2)

    CC, SS = np.where(r_diff < tolerance)

    return CC

def activity_var_selection(dff_data, display = True):
    '''
    select the high-activity cells based on the variance of the Delta f/f.
    Score the activity for each neuron, try to find possible pattern.
    Return the score list and estimated cut-off 
    '''
