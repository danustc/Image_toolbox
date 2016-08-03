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
    
    ofst = 1
    for im_name in im_list:
        name_base = im_name[:-4]
        new_stack = np.copy(tifffunc.read_tiff(name_base))
        Drift_C = Drift_correction(new_stack, mfit = 7)
        a_stack = Drift_C.drift_correct(offset = ofst)
        name_out = name_base+'_a'
        tifffunc.write_tiff(a_stack[ofst:].astype('uint16'), name_out)

    """
    raw_image = 'img_aligned'
    im_stack = tifffunc.read_tiff(raw_image).astype('float64')
    CE = Cell_extract(im_stack)
    CE.stack_blobs()
    print(CE.bl_flag)
    n_frame = 10
    data_slice = CE.image_signal_integ(n_frame)
    
    
    
    
if __name__ == '__main__':
    main()

