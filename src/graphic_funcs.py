"""
Created by Dan on 08/16/16. 
This one contains all the plt-based graphic functions shared among all the functions.
"""

import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------Some tiny functions -------------------------------------

def scale_bar(fig_ax, pxl_size = 0.295, sc_length = 20):
    """
    fig: current figure handle
    question: should I pass a figure or a figure axes? 
    pxl_size: the pixel size of the image, unit: micron
    sc_length: scale bar length, unit micron.
    """


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
    OMG this works!!!! :D
    """
    def __init__(self, plt_draw):
        self.plt_draw = plt_draw
        self.cid = plt_draw.figure.canvas.mpl_connect('button_press_event', self)
        
        
    def __call__(self, event):
        # the question is, how to catch up this?
        if event.inaxes!=self.plt_draw.axes: 
            return
        
        print("indices:",np.uint16([event.ydata, event.xdata]))
        
        self.xc = event.xdata
        self.yc = event.ydata