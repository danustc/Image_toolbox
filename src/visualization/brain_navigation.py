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
        ax = self.figs.add_subplot(111, projection = '3d')
    # ----------------------Below are property members and their setters

    @property
    def coord(self):
        return self._coord
    @coord.setter
    def coord(self, new_coord):
        self._coord = new_coord

    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self, new_signal):
        self._signal = new_signal

    @property
    def refim(self):
        return self._refim
    @refim.setter
    def refim(self, new_refim):
        self._refim = new_refim

    @property
    def pxl(self):
        return self._pxl
    @pxl.setter
    def pxl(self, new_pxsize):
        self._pxl = new_pxsize

    # -------------------Below are visualization functions --------------

    def show_cell(self, ind_cell):
        '''
        ind_cell: the index of the cells, can be a number or a list/array.
        perspec: the angle of view, currently only z, y, x are provided.
        only displays the image and the position of the cells
        '''
        ax = self.figs.axes[0]

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
        ax = self.figs.axes[0]
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


    def fig_save(self,fig_path):
        self.figs.savefig(fig_path)


# ---------------------------Below is the interactive mode 


def onclick_coord(event):
    """
    hand select coordinate on a figure and return it.
    adapted from stackoverflow.
    """
    print(event.x, event.y, event.xdata, event.ydata)


# ---------------------------------Next, some awkward classes -------------------------------

class coord_click():
    """
    Purpose: return the coordinates of mouse click, given an axes.
    OMG this works!!!! :D But how can I return self.xc and self.yc?
    """
    def __init__(self, plt_draw):
        self.plt_draw = plt_draw
        self.cid = plt_draw.figure.canvas.mpl_connect('button_press_event', self)
        self.coord_list = []


    def __call__(self, event):
        # the question is, how to catch up this?
        if event.inaxes!=self.plt_draw.axes:
            return

        print("indices:",np.uint16([event.ydata, event.xdata]))

        self.xc = event.xdata
        self.yc = event.ydata
        self.coord_list.append([self.yc, self.xc])

    def catch_values(self):
        """
        give the value in self.xc, yc to the outside handle.
        """
        coord = np.array(self.coord_list)
        # ---- clear
        self.xc = None
        self.yc = None
        return coord
