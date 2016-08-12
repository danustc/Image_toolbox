"""
Updated by Dan on 08/11/2016.
"""

import matplotlib.pyplot as plt
import tifffunc
import numpy as np
import os
import glob
from Preprocess import Deblur, Drift_correction
from Cell_extract import Cell_extract


def main():
    """
    hroot = '/Public/Zebrafish_ispim/'
    abspath = os.path.abspath(hroot)
    aq_date = '2016-03-21/'
    fd = abspath + aq_date
        
    im_list = glob.glob(fd+ '/*TP_*.tif') # list all the tiff files in the folder 
    
    ofst = 1re 
    for im_name in im_list:
        name_base = im_name[:-4]
        new_stack = np.copy(tifffunc.read_tiff(name_base))
        Drift_C = Drift_correction(new_stack, mfit = 7)
        a_stack = Drift_C.drift_correct(offset = ofst)
        name_out = name_base+'_a'
        tifffunc.write_tiff(a_stack[ofst:].astype('uint16'), name_out)

    """
    raw_image = 'raw_image'
    dbl_image = 'raw_image_deblur'
    DB = Deblur(raw_image)
    n_slice = 12
    
    im_raw = DB.raw_stack[n_slice]
    
    im0, ifilt = DB.image_high_trunc_inplane(n_slice)
    
    fig = plt.figure(figsize=(6,5))
    
    plt.imshow(ifilt, cmap='Greys_r')
    plt.savefig('LPF')
    plt.clf()
    
    plt.imshow(im0, cmap = 'Greys_r')
    plt.savefig('HPF')
    plt.clf()
    
    plt.imshow(im_raw, cmap = 'Greys_r')
    plt.savefig('RAW')
    plt.clf()
      
    dbl_stack = tifffunc.read_tiff(dbl_image).astype('float64')
    CE_dbl = Cell_extract(dbl_stack)
    CE_dbl.stack_blobs(diam = 6)
    print(CE_dbl.bl_flag)
#     CE_dbl.stack_signal_archive()
#     CE_dbl.save_archive('arc_img_dbl')
#     n_frame = 10    
#     data_slice_2 = CE_dbl.image_signal_integ(n_frame)
#     fig2 = CE_dbl.frame_display(n_frame, False)
#     fig2.savefig('dbl_s10')
    fig3 = CE_dbl.volume_display(3.00)
    plt.show()
#     
#     fig3.savefig('stack_reconstruction')

if __name__ == '__main__':
    main()

