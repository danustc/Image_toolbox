'''
Created on 04/25/2017 by Dan.
Last update: 04/25/2017
Function: navigate a stack of images and display cells across the stack.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt


def slice_display(slice_blobs, ref_image = None):
    '''
    slice_blobs: a 2-D array specifying the cell coordinates
    ref_image: a 2-D array specifying the reference image.
    '''
    if ref_image is None:
        ymax = np.max(slice_blobs[:,0])
        xmax = np.max(slice_blobs[:,1])
        figd = plt.figure(figure = (6., 6.*ymax/xmax))
    else:
        ny, nx = ref_image.shape
        figd = plt.figure(figsize = (6., 6.*ny/nx))
        ax = figd.add_subplot(111)
        ax.imshow(ref_imagei, cmap = 'Greys_r')
        ax.set_xlim([0,nx])
        ax.set_ylim([0,ny])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    ax.scatter(slice_blobs[:,1], slice_blobs[:,0], c='g', s = 6)

    return figd
