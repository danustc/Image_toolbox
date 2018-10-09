"""
Created by Dan on 08/16/16.
This one contains all the plt-based graphic functions shared among all the functions.
Last update: 09/30/18
"""

import numpy as np
import matplotlib.pyplot as plt

def mip_stack(img_stack, rot_axis = 'x', rot_angle = 0):
    '''
    create a max intensity projection of the input image stack.
    '''
    NZ, NY, NX = img_stack.shape
    if rot_angle == 0:
        return img_stack.max(axis = 0)


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



def slice_display(slice_blobs, title = None, ref_image = None, s_diag = 15):
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
            ax.scatter(fr[:,1], fr[:,0], edgecolors = global_ccode[ii], facecolors = 'none', s = s_diag)
            ii+=1
    else:
        ax.scatter(slice_blobs[:,1], slice_blobs[:,0], c='g', s = s_diag, facecolors = 'none')

    ax.set_title(title, fontsize = 14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    figd.tight_layout()
    return figd


def slices_compare(im_slice1, im_slice2, title_1 = None, title_2 = None, fh = 3.):
    '''
    imshow two slices side by side.
    '''
    y1, x1 = im_slice1.shape
    y2, x2 = im_slice2.shape

    figc = plt.figure(figsize = (2*(x1+x2)/(y1+y2)*fh, fh))
    ax1 = figc.add_subplot(121)
    ax2 = figc.add_subplot(122)
    ax1.imshow(im_slice1, cmap = 'Greys_r')
    ax1.set_xlim([0, x1])
    ax1.set_ylim([0, y1])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    if title_1 is not None:
        ax1.set_title(title_1, fontsize = 13)

    ax2.imshow(im_slice2, cmap = 'Greys_r')
    ax2.set_xlim([0, x2])
    ax2.set_ylim([0, y2])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    if title_2 is not None:
        ax2.set_title(title_2, fontsize = 13)

    figc.tight_layout()
    return figc



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


