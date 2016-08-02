from Preprocess import Preprocess, Drift_correction
import matplotlib.pyplot as plt
import tifffunc
import numpy as np
import os
import Cell_extract as CE





def main():
    # remove the comment from the line below: 
#     datapath = 'C:\username\Documents\......(fill it up)'

    # step 1: do the image deblurring 
    datapath = ''
    input_name = 'raw_image_deblur' # replace this with your image name.
    new_stack = tifffunc.read_tiff(input_name)
    
    
    
    Drift_C = Drift_correction(new_stack, mfit=3)
    a_stack = Drift_C.drift_correct()
    
    output_name = datapath + input_name  + '_aligned'
    tifffunc.write_tiff(a_stack.astype('uint16'), output_name)


    


    # next, let's add some cell extraction procedures.
    
#     new_slice = tifffunc.read_tiff(output_name).astype('float')
#     blobs_list = CE.image_blobs(new_slice, blob_set=[5,3,10], th = 120., OL=1.)
#     print("Extracted cells:", len(blobs_list))
#     print(blobs_list)

if __name__ == '__main__':
    main()

