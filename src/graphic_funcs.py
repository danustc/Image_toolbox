"""
Created by Dan on 08/16/16. 
This one contains all the plt-based graphic functions shared among all the functions.
Last update: 08/19/16
"""

import matplotlib.pyplot as plt

import numpy as np
# ---------------------------------Some tiny functions -------------------------------------

def image_scale_bar(fig_im, location, sc_length = 20., pxl_size = 0.295):
    """
    fig: current figure handle
    question: should I pass a figure or a figure axes? 
    pxl_size: the pixel size of the image, unit: micron
    location: the center of the scale bar. unit: px
    sc_length: scale bar length, unit micron.
    default width of the scale bar: 10 px
    """
    ax = fig_im.get_axes()[0]
    h_sc = 0.5*sc_length/pxl_size
    
    xs = [location[1] - h_sc, location[1]+ h_sc]
    ys = [location[0], location[0]]
    ax.plot(xs, ys, '-w', linewidth = 10)
    # done with image_scale_bar
    


def image_zoom_frame(fig_im, c_nw, c_se, cl = 'w'): 
    """
    frame a rectangular area out from an imshow() image. default: white dashed line
    OK this works.
    """
    ax = fig_im.get_axes()[0]
    y1, x1 = c_nw # northwest corner coordinate 
    y2, x2 = c_se # southeast corner coordinate
    ax.plot([x1,x1], [y1, y2], '--', color = cl)
    ax.plot([x1,x2], [y1, y1], '--', color = cl)
    ax.plot([x2,x2], [y1, y2], '--', color = cl)
    ax.plot([x1,x2], [y2, y2], '--', color = cl)
    # done with image_zoom_frame


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
#--------------------------------------Done with coord click 


