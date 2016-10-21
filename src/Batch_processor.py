"""
Created by Dan on 08/02/16
For group processing
Last update: 09/13/16
"""

import glob
import tifffunc
import numpy as np
from Preprocess import Drift_correction
from Cell_extract import Cell_extract
from Postprocess import brain_construct
import matplotlib.pyplot as plt
    
def group_postprocess(datapath, dims):
    """
    """
    BC = brain_construct(datapath, dims)
    BC.zs_construct()
    BC.corrmap_stack()
    red_list = BC.red_detect(d_crt = 2.5, c_thresh = 0.6)
    r1, r2 = red_list['001']
    c1 = BC.z_images[1].coord
    c2 = BC.z_images[2].coord
    
    
#     np.savez(datapath + 'redlist', **red_list)
    
    im1 = BC.z_stack[1]
    im2 = BC.z_stack[2]
    
    dims = im1.shape
    fig = plt.figure(figsize=(9,4))
    plt.axis('off')
    ax1 = fig.add_subplot(121)
    ax1.imshow(im1, cmap='Greys_r')
    for k1 in r1:
        y,x = c1[k1]
        c = plt.Circle((x,y), 6, color='r', linewidth=1, fill=True)
        ax1.add_patch(c)
    plt.axis('off')
    plt.tight_layout()
    
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(im2, cmap='Greys_r')
    for k2 in r2:
        y,x = c2[k2]
        c = plt.Circle((x,y), 6, color='r', linewidth=1, fill=True)
        ax2.add_patch(c)
    plt.axis('off')
    plt.tight_layout()
    
    
    
    plt.tight_layout()
    plt.savefig(datapath+'red_ex')
    
    
    
    
    corr_map = BC.corr_map
    dist_map = BC.dist_map
    return corr_map, dist_map



def group_alignment(datapath, nameflag = 'TS', ofst = 1, mfit = 7):
    # align all th
    # prerequisite: all slices should be deblurred inplane
    tiff_list = glob.glob(datapath + '*' + nameflag+'*.tif')
    for tiff_name in tiff_list: 
        n_base = tiff_name[:-4] # remove '.tif' extension
        new_stack = np.copy(tifffunc.read_tiff(n_base))
        Drift_C = Drift_correction(new_stack, mfit)
        a_stack = Drift_C.drift_correct(offset = ofst)
        n_out = n_base+'_a'
        tifffunc.write_tiff(a_stack[ofst:].astype('uint16'), n_out)
        

            

def group_cell_extract(datapath, name_flag = 'TS*'):
    stack_list = glob.glob(datapath+'*'+ name_flag + '*.tif')
    for stk_name in stack_list:
        n_base = stk_name[:-4]
        im_stack = tifffunc.read_tiff(n_base)
        CE = Cell_extract(im_stack)
        CE.stack_blobs()
        