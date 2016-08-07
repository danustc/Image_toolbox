"""
Updated by Dan on 08/03/2016.
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
    dbl_image = 'raw_image_deblur'
    
    

    
    dbl_stack = tifffunc.read_tiff(dbl_image).astype('float64')
    CE_dbl = Cell_extract(dbl_stack)
    CE_dbl.stack_blobs(diam = 6)
    print(CE_dbl.bl_flag)
    
#     data_slice_1 = CE_raw.image_signal_integ(n_frame)
#     print(data_slice_1.shape)
#     data_slice_2 = CE_dbl.image_signal_integ(n_frame)
#     print(data_slice_2.shape)
#     fig1 = CE_raw.frame_display(n_frame)
#     fig1.savefig('raw_s10')
#     fig2.savefig('dbl_s10')
    fig3 = CE_dbl.volume_display(3.00)
    fig3.show()


if __name__ == '__main__':
    main()

