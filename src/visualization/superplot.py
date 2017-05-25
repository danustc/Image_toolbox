'''
Created by Dan on 05/19/2017. For imaging display.
'''

import numpy as np
import matplotlib.pyplot as plt

def superplot(im_array, nrow, ncol, as_ratio = 1., cmaps = None, cmap_group = None, ax_num = False, row_labels = None, col_labels = None, padding = [0.05, 0.05]):
    '''
    im_array: an array of images, all in 2D arrays, can be different in size. Assume that the order of the array is arranged by rows, i.e., I[0][0], I[0][1] ... I[0][N]; I[1][0], I[1][1], ... I[1][N]; ... I[M][N].
    nrow: the number of rows in the subplot array
    ncol: the number of columns in the subplot array
    as_ratio: aspect ratio of each subplot
    cmaps: a list of cmaps. If the number of cmaps matches the number of subplots, then each subplot has an individual color map
    cmap_group: 0 --- The rows share the same color map
                1 --- The columns share the same color map
                None --- each subplot is coded independently.
    ax_num: whether the axis numbers should be shown or not.
    row_labels: Add the row labels on the very left
    col_labels: Add the col_labels on top of the plots
    padding: the w space and h space between subfigures
    '''


    nplots = len(im_array)
    if(nrow * ncol ~=nplots):
        print("The grid array and the image array size mismatch! ")
        return None
    else:
        fw = 2.0*ncol # the figure width
        fh = 2.0*nrow*as_ratio # the figure height
        fig_pc, axes = plt.subplots(nrow, ncol, figsize = (fw,fh))
        if cmaps is None:
            cmaps = 'F'



        fig_pc.subplots_adjust(hspace = padding[0], wspace = padding[1]) # change the padding
        for nr in range(nrow): # iterate through rows

            for nc in range(ncol): # iterate through columns
