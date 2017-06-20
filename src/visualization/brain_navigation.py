'''
Created on 04/25/2017 by Dan.
Last update: 04/26/2017
Function: navigate a stack of images and display cells across the stack.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.stack_operations import stack_section
from mpl_toolkits.mplot3d import Axes3D

global_ccode = 'grbmcyk'
px_size = 0.295 # pixel size: 0.295 microns

def slice_display(slice_blobs, title = None, ref_image = None):
    '''
    slice_blobs: a 2-D array specifying the cell coordinates
    ref_image: a 2-D array specifying the reference image.
    Adding multi-slice plotting:
    '''
    if ref_image is None:
        figd = plt.figure(figsize= (6.,4.) )
        ax = figd.add_subplot(111)
    else:
        ny, nx = ref_image.shape
        figd = plt.figure(figsize = (6., 6.*ny/nx))
        ax = figd.add_subplot(111)
        ax.imshow(ref_image, cmap = 'Greys_r')
        ax.set_xlim([0,nx])
        ax.set_ylim([0,ny])

    if isinstance(slice_blobs, list):
        ii = 0
        NS = len(slice_blobs)
        for fr in slice_blobs:
            ax.scatter(fr[:,1], fr[:,0], c = global_ccode[ii], s = 20*(NS-ii))
            ii+=1
    else:
        ax.scatter(slice_blobs[:,1], slice_blobs[:,0], c='g', s = 10)

    ax.set_title(title, fontsize = 14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    figd.tight_layout()
    return figd


def slices_compare(im_slice1, im_slice2):
    '''
    imshow two slices side by side.
    '''
    y1, x1 = im_slice1.shape
    y2, x2 = im_slice2.shape

    figc = plt.figure(figsize = ((x1+x2)/(y1+y2)*6., 6.))
    ax1 = figc.add_subplot(121)
    ax2 = figc.add_subplot(122)
    ax1.imshow(im_slice1, cmap = 'Greys_r')
    ax1.set_xlim([0, x1])
    ax1.set_ylim([0, y1])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.imshow(im_slice2, cmap = 'Greys_r')
    ax2.set_xlim([0, x2])
    ax2.set_ylim([0, y2])
    ax2.get_xaxis().set_visible(False)

    figc.tight_layout()
    return figc
    ax2.get_yaxis().set_visible(False)



def stack_display(zstack_3d, cl = 'b'):
    '''
    display a z-distribution of a stack
    column conventions: 0 -- z; 1 -- y; 2 -- x
    '''
    zs, ys, xs = zstack_3d[:,0], zstack_3d[:,1], zstack_3d[:,2]
    fig3d = plt.figure(figsize= (10,10))
    ax = fig3d.add_subplot(111, projection = '3d')
    ax.scatter(xs, ys, zs, c = cl, depthshade = True)
    ax.set_xlabel('Anterior -- Posterior', fontsize = 14)
    ax.set_ylabel('Left -- Right', fontsize = 14)
    ax.set_zlabel('Ventral--Dorsal', fontsize = 14)
    fig3d.tight_layout()
    return fig3d



# ------------------------Classes------------------------
class region_view(object):
    '''
    Visualize a brain region and navigate through it.
    Data to read: a ZD stack and a dictionary containing coord && dff.
    '''
    def __init__(self, coord = None, signal = None, ZD_image = None, pxl_size = 0.295):
        '''
        data_pk:    'coord' -- coordinates
                    'data'  -- fluorescence signal (raw_f)
                    'dff'   -- calculated DeltaF/F
        '''
        self._coord = None
        self._signal = None
        self._refim = None
        self._pxl = None

        if coord is not None:
            self.coord = coord
        if signal is not None:
            self.signal = signal
        if ZD_image is not None:
            self.refim = ZD_image
        self.pxl = pxl_size

        self.figs = plt.figure()
        ax = self.figs.add_subplot(111)
    # ----------------------Below are property members and their setters

    @property
    def coord(self):
        return self._coord
    @coord.setter(self, new_coord):
        self._coord = new_coord

    @property
    def signal(self):
        return self._signal
    @signal.setter(self, new_signal):
        self._signal = new_signal

    @property
    def refim(self):
        return self._refim
    @refim.setter(self, new_refim):
        self._refim = new_refim

    @property
    def pxl(self):
        return self._pxl
    @pxl.setter(self, new_pxsize):
        self._pxl = new_pxsize

    # -------------------Below are visualization functions --------------

    def show_cell(self, ind_cell):
        '''
        ind_cell: the index of the cells, can be a number or a list/array.
        perspec: the angle of view, currently only z, y, x are provided.
        only displays the image and the position of the cells
        '''
        ax = self.figs.gca()

        if(np.isscalar(ind_cell)):
            z,y,x= self.coord[ind_cell,:]
            ax.scatter(x*self.pxl, y*self.pxl, z, c = 'g', s = 10)
        else:
            coords = self.coord[ind_cell, :]
            ax.scatter(coords[:,2]*self.pxl, coords[:,1]*self.pxl, coords[:,0], c = 'g', s = 10)
        # done with show cell


    def show_grey_slice(self, slice_position, view = 'z'):
        '''
        slice:position: the position where the z-stack should be sectioned.
        view: the angle of view. must be z or x, y.
        '''
        ax = self.figs.gca()
        if(view == 'x' or view =='y'):
            pxl_position = self.pxl*slice_position
        else:
            pxl_position = slice_position
        im_section = stack_section(self.refim, pxl_position, view)
        ax.imshow(im_section, cmap = 'Greys_r')
        # done with show_grey_slice

    def fig_resize(self, new_size):
        '''
        resize figure
        '''
        self.figs.set_size_inches(new_size)

