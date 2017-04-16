"""
Recreated by Dan on 10/10/2016.
Test redundancy removal.
"""


import os
from src.pipeline.z_dense import z_dense_ref, z_dense_construct
from src.networks.group_selection import cell_selection
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import src.pipeline.tifffunc as tifffunc
import src.pipeline.Background_correction as Background_correction
import src.pipeline.Cell_extract as Cell_extract
import src.dynamics.df_f as df_f
from src.shared_funcs.numeric_funcs import circ_mask_patch
from src.shared_funcs.graphic_funcs import slice_display

def dumb1():
    npz_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/Habenula/Oct25_B3_TS18.npz'
    im_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/Habenula/Oct25_B3_TS18.tif'
    sample_stack = tifffunc.read_tiff(im_path).astype('float64')

    coord_s = np.array([[120,77], [92,153], [98,92], [207,49], [179, 165], [133, 141], [126,98], [442, 77],[581, 96], [527, 66], [440,127], [511,153], [584,140]])

    data_raw = np.load(npz_path)
    data_FB = data_raw['data']
    coord_FB = data_raw['xy']

    cfind = cell_selection(coord_FB, coord_s, 10)
    print(cfind)

    f_raw = data_FB[:,cfind]
    fig_im = slice_display(sample_stack[1])
    ax = fig_im.add_subplot(111)
    ax = fig_im.gca()
    ax.scatter(coord_FB[cfind,1], coord_FB[cfind,0], facecolors = 'g', edgecolors = 'g')
    for ii in np.arange(len(cfind)):
        ax.text(coord_FB[cfind[ii], 1], coord_FB[cfind[ii], 0], str(ii+1), color = 'y')

    ax.text(50,50, 'lHb', fontsize = 18, color = 'w')
    ax.text(50,600, 'rHb', fontsize = 18, color = 'w')
    ax.set_xlim([0,750])
    ax.set_ylim([0,640])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig_im.gca().invert_yaxis()
    fig_im.gca().invert_xaxis()
    fig_im.tight_layout()
    fig_im.savefig('slice.png')


    dff, f_base = df_f.dff_raw(f_raw, ft_width = 3)
    dff[dff<0] = 0.0
    fm = f_base.max()
    dff_fine = np.zeros_like(dff)
    print(dff.size)
    for ii in np.arange(dff.shape[1]):
        dff_fine[:,ii], wd = df_f.dff_expfilt(dff[:,ii],dt = 0.50, t_width = 1.0)

    fig = df_f.nature_style_dffplot(dff[20:], 0.5, 0.20)

    fig.savefig('dff_raw')
    plt.clf()
    plt.plot(wd)
    plt.savefig('wd')

#     dmap = BC.dist_map

    DB = Background_correction.Deblur(sample_stack[1:20])
    DB.stack_high_trunc()
    dbl_stack = DB.get_stack()
    N_select = 0

    print(sample_stack.dtype)
    print(dbl_stack.dtype)
    slice_raw = sample_stack[N_select]
    slice_dbl = dbl_stack[N_select]

    CE_raw = Cell_extract.Cell_extract(sample_stack)
    CE_dbl = Cell_extract.Cell_extract(dbl_stack)

    data_raw = CE_raw.image_blobs(N_select)
    data_dbl = CE_dbl.image_blobs(N_select)

    NY, NX = slice_raw.shape
    numraw, numdbl = (data_raw.shape[0], data_dbl.shape[0])

    fig_raw = plt.figure(figsize = (10,4.5))
    ax1 = fig_raw.add_subplot(121)
    ax1.imshow(slice_raw, cmap = 'Greys_r')
    ax1.text(50, 50, 'lHb', color = 'y')
    ax1.text(50, 600, 'rHb', color = 'y')
    ax1.set_xlim([0, NX-5])
    ax1.set_ylim([0, NY-5])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.set_title('Before background subtraction')

    ax2 = fig_raw.add_subplot(122)
    ax2.imshow(slice_raw, cmap = 'Greys_r')
    ax2.scatter(data_raw[:,1], data_raw[:,0], facecolors = 'none', edgecolors = 'g')
    ax2.set_xlim([0, NX-5])
    ax2.set_ylim([0, NY-5])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()
    ax2.invert_xaxis()

    ax2.set_title("Detected blobs: " + str(numraw))
    fig_raw.tight_layout()
    fig_raw.savefig('raw_6dpf.png')

    fig_dbl = plt.figure(figsize=(10, 4.5))
    ax1 = fig_dbl.add_subplot(121)
    ax1.imshow(slice_dbl, cmap = 'Greys_r')
    ax1.set_xlim([0, NX-5])
    ax1.set_ylim([0, NY])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    ax1.text(50, 50, 'lHb', color = 'y')
    ax1.text(50, 600, 'rHb', color = 'y')

    ax1.set_title("After background subtraction")


    ax2 = fig_dbl.add_subplot(122)
    ax2.imshow(slice_dbl, cmap = 'Greys_r')
    ax2.scatter(data_dbl[:,1], data_dbl[:,0], facecolors = 'none', edgecolors = 'g')
    ax2.set_xlim([0, NX-5])
    ax2.set_ylim([0, NY])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    ax2.set_title("Detected blobs: " + str(numdbl))
    fig_dbl.tight_layout()
    fig_dbl.savefig('dbl_6dpf.png')




if __name__ == '__main__':
    dumb1()
