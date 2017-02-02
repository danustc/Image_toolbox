'''
Created by Dan on 02/01/2017
'''

import numpy as np

def cell_selection(coord_data, coordinates, tolerance = 5):
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
