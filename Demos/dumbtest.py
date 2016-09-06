"""
Updated by Dan on 08//30/2016.
"""
import sys
sys.path.insert(0, '../src')
import matplotlib.pyplot as plt
import tifffunc
import numpy as np
import os
import time
import glob
from Preprocess import Deblur, Drift_correction
from Cell_extract import Cell_extract
from Pipeline import pipeline_zstacks, pipeline_tstacks
from Dynamics import Temporal_analysis
from graphic_funcs import coord_click
from df_f import nature_style_dffplot, dff_expfilt
from graphic_funcs import image_zoom_frame
from drift_correct_1D import DC_1D




def dumb1():
    """
    This program tests pipeline_zstacks 
    """
    

def dumb2():

    """
    OK this part already works. 
    """ 
#     DB_gaussian = Deblur('sample_tstack.tif', sig = 25, ftype = 'g')
#     DB_uniform = Deblur('sample_tstack.tif', sig = 25, ftype = 'u')
# #     
#     print("initialized.")
# #     
#     db_g = DB_gaussian.stack_high_trunc()
#     db_u = DB_uniform.stack_high_trunc()
# #     
#     print("deblurred.")
# #     
#     DC_gaussian = Drift_correction(db_g, mfit = 0)
#     DC_uniform = Drift_correction(db_u, mfit = 0)
# #     
#     print("Drift correction initialized.")
# #     
# #     
#     nstack_g = DC_gaussian.drift_correct(offset = 0, ref_first = True)
#     nstack_u = DC_uniform.drift_correct(offset = 0, ref_first = True)
# # 
#     print("Drift correction completed.")
# #
# #
#     dbstack_gs = tifffunc.read_tiff('ts_gaussian_s25.tif')
    refim = tifffunc.read_tiff('ref_im.tif').copy()
#     tifffunc.write_tiff(dbstack_gs[0].astype('uint16'), 'ref_im.tif')
#     tifffunc.write_tiff(nstack_u.astype('uint16'), 'ts_uniform_s25.tif')
# # 
# #     
#     CE_gaussian = Cell_extract(nstack_g)
#     CE_uniform = Cell_extract(nstack_u)
# #     
#     blob_stack_gaussian = CE_gaussian.stack_signal_propagate(0)
#     blob_stack_uniform = CE_uniform.stack_signal_propagate(0)
# # 
#     np.savez('bs_gaussian_s25', **blob_stack_gaussian)
#     np.savez('bs_uniform_s25', **blob_stack_uniform)
# #     
    dims = [584, 888]
    data_gaussian = np.load('bs_gaussian.npz')
    data_uniform = np.load('bs_uniform.npz')
    
    TD_gaussian = Temporal_analysis(data_gaussian, dims, refim)
    TD_uniform = Temporal_analysis(data_uniform, dims, refim)

    n_gauss = TD_gaussian.n_cell
    n_uni = TD_uniform.n_cell
    
    print(n_gauss, n_uni)


    dff_gauss = TD_gaussian.dff_calculation(ft_width = 10)
    dff_unif = TD_uniform.dff_calculation(ft_width = 10)
    
    kcenter = np.array([266,393])
    k_select_gauss = TD_gaussian.signal_profile_single(kcenter, rad = 40, sel = 3)
    k_select_unif = TD_uniform.signal_profile_single(kcenter, rad = 40, sel = 3)


    fig = plt.figure(figsize = (9,7))
    
    sel = len(k_select_gauss)
    TD_gaussian.cell_show()
#     TD_uniform.cell_show()
    
      
    for nsel in np.arange(sel):
        s_mark = k_select_gauss[nsel]
        mark = TD_gaussian.coord[s_mark]
        tx = str(nsel+1)
        TD_gaussian.cell_mark(mark, tx, cl = 'y')
      
    
    
    
    tt = np.arange(100)*0.8
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(tt, dff_gauss[:,k_select_gauss])
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(tt, dff_unif[:, k_select_unif])
    
    
    plt.tight_layout()
    plt.savefig('Whole_select')



#     ax1 = fig.add_subplot(121)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#     DC = Drift_correction(db_stack, mfit = 30)
#     new_stack = DC.drift_correct(offset=498, ref_first=True).astype('uint16')
#     ax2 = fig.add_subplot(122)
#     ax2.imshow(np.log(DC.xcorr), cmap = 'Greys_r', interpolation = 'none')
#     plt.show()
                                                                                                                                                                                                                                                                                                                                          
        
def dumb3():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    """
    suppose drift already corrected. Now time for real cell extraction! 
    The first part is almost identical to dumb2().
    
    """
    hroot = 'D:\Data/'
    abspath = os.path.abspath(hroot)                                                                                                        
    aq_date = '/2016-03-21/zp_A2_4_TS1\\'
    work_folder = abspath + aq_date+'\\'
    
    
    
    print(work_folder)
    t_pipeline = pipeline_tstacks(work_folder)
    t_pipeline.tstack_zseries(deblur=30, align = True)
    
    
    
    
def dumb4():
    """    
    test dynamics
    """
    tt = np.arange(400)*0.80 
    TS_data = np.load('A24TS1ZPdbal_005.npy')

    refstack = tifffunc.read_tiff('refim.tif')
    
    ref_im = refstack[0]
    
    
    dims = [760, 832]
    TD=Temporal_analysis(TS_data, dims, ref_im=ref_im)
    dff = TD.dff_calculation(ft_width = 6)
    kf = np.array([0.11,0.16])
    
    kpros, k_select = TD.firing_analysis(sfreq = 1.25, kfrac = kf, k_threshold = 1.95)
    
    fig = plt.figure(figsize = (8,7))
    
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(kpros[:,0]/0.625, kpros[:,1:])
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(tt, dff[:, k_select])
    
    df_try = dff[:, k_select[2]]
    df_filt, wd = dff_expfilt(df_try, dt=0.80, t_width = 1.2)
    ax2.plot(tt, df_filt+2.50, '-g', linewidth =2)
    ax2.plot(tt, df_try+2.50, '-m')
#     plt.plot(dff[:,26:29])
    plt.show()
    plt.clf()
    plt.plot(wd)
    plt.show()
    
    
#    
#     
    

def dumb5():
    raw_stack = tifffunc.read_tiff('ZP0.tif').astype('float')
    CE = Cell_extract(raw_stack)
    CE.stack_blobs(msg = False)
    
#     DC = DC_1D(raw_stack)
#     sx, sy = DC_1D.drift_correct()
  
  
  

if __name__ == '__main__':
    start_time = time.time()
    dumb2()

    print("--- %s seconds ---" % (time.time() - start_time))