"""
Updated by Dan on 08//30/2016.
"""
import sys
from numeric_funcs import histo_peak
sys.path.insert(0, '../src')
import matplotlib.pyplot as plt
import tifffunc
import tifffile
import numpy as np
import os
import time
import glob
from Preprocess import Deblur, Drift_correction
from Cell_extract import Cell_extract
from Pipeline import pipeline_zstacks, pipeline_tstacks
from df_f import nature_style_dffplot, dff_expfilt
from graphic_funcs import image_zoom_frame
from Batch_processor import group_postprocess
from Postprocess import z_image
from scipy.ndimage import gaussian_filter, uniform_filter
from Dynamics import Temporal_analysis
import matplotlib 

def dumb1():
    """
    This program tests pipeline_zstacks 
    """
    dph = '/home/sillycat/Programming/Python/Image_toolbox/data_test/short/'
    zlist = glob.glob(dph+'*00*.npz*')
    zset = np.load(zlist[2])
    coord = zset['xy']
    TA = Temporal_analysis(zset, dims = [740, 880])
    TA.dff_calculation(ft_width = 4)
    kp, ks = TA.firing_analysis(sfreq = 1.25, kfrac = np.array([0.80, 0.90]), k_threshold = 0.80)
    
    TA.cell_show()
    N_select = -1
    
    k_select = TA.dff[:,ks]
    fig_sk = plt.figure(figsize=(7,9))
    ax_sk = fig_sk.add_subplot(111)
    tt = 0.8*np.arange(400)
    
    ii=0
    for s in ks:
        mark = coord[s]
        fig_mk = TA.cell_mark(mark, tx = str(ii+1), dr=5, cl = 'r')
        ax_sk.plot(tt, TA.dff[:,s]+ ii*0.55)
        ii+=1
    
    font = {'size'   : 14}

    matplotlib.rc('font', **font)
    
    ax_sk.get_yaxis().set_visible(False)
    ax_sk.set_xlim([0, 320])
    ax_sk.set_ylim([-0.4, ii*0.55])
    ax_sk.set_xlabel('time (s)', fontsize = 14)
    plt.tight_layout()
    fig_sk.savefig('dff_mark_hf')
    
    fig_mk.savefig('cell_mark_hf')
    
    
    
    plt.plot(k_select)
    plt.savefig('TF_select_hf')

    fig_nature = nature_style_dffplot(k_select, dt = 0.8, sc_bar = 0.20)
    fig_nature.savefig('dff_nature_hf')
    
        
def dumb3():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    """
    suppose drift already corrected. Now time for real cell extraction! 
    The first part is almost identical to dumb2().
    """
    
    dph = '/home/sillycat/Programming/Python/Image_toolbox/Demos/MB.tif'
    
#     Deblur(dph,sig = 30)


    image = tifffunc.read_tiff(dph)
    zim_i = image[0]
    zim_m = image[50]
    zim_f = image[-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(zim_i, cmap = 'Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('MB_init')
    plt.clf()
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(zim_m, cmap = 'Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('MB_mid')
    plt.clf()
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(zim_f, cmap = 'Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('MB_final')
    plt.clf()
    

    
    
    
#     PZP = pipeline_zstacks(dph, tpflags='ref',offset = 0)
#     PZP.zstack_tseries(deblur = 30, align = True)
    
    
    
def dumb4():
    """    
    test dynamics
    """
    dph = '/home/sillycat/Programming/Python/Image_toolbox/Demos/refim.tif'
    image = tifffunc.read_tiff(dph)
    pv = np.zeros(image.shape[0])
    
    for ii in np.arange(image.shape[0]):
        #def histo_peak(im_arr, val_cut, nbin = 50, ext = 1):
        zim = image[ii]
        vc = zim.mean()*0.10
        pv[ii] = histo_peak(zim, val_cut = vc, nbin = 200, ext = 1)
#         plt.hist(zim.flatten(), bins = 200, facecolor = 'g', alpha = 0.60)#     
#         plt.savefig('hist_'+str(ii))
#         plt.tight_layout()
#         plt.clf()

    plt.plot(pv)
    plt.savefig('hist_background')
    

def dumb5():
    raw_stack = tifffunc.read_tiff('ZP0.tif').astype('float')
    CE = Cell_extract(raw_stack)
    CE.stack_blobs(msg = False)
    
#     DC = DC_1D(raw_stack)
#     sx, sy = DC_1D.drift_correct()
  
  
  

if __name__ == '__main__':
    start_time = time.time()
    dumb1()

    print("--- %s seconds ---" % (time.time() - start_time))