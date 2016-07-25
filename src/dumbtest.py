from Preprocess import Preprocess
import matplotlib.pyplot as plt
import tifffunc
import numpy as np
import Cell_extract as CE


def main():
    # remove the comment from the line below: 
#     datapath = 'C:\username\Documents\......(fill it up)'

    # step 1: do the image deblurring 
    datapath = ''
    input_name = 'test_image' # replace this with your image name.
    sig = 30 # play with this number. Hint: create a for loop like:
    nslice = 10 # you can choose the best-looking slice in the whole stack
#     for sig in np.arange(10, 40, step = 5):
    impp = Preprocess(datapath + input_name, sig)
    new_slice = impp.image_high_trunc_inplane(nslice) # not doing adjacent corrections 
    output_name = datapath + input_name + 'w_'+ str(sig) + '_s' + str(nslice)
    tifffunc.write_tiff(new_slice.astype('uint16'), output_name)


    # next, let's add some cell extraction procedures.
    
#     new_slice = tifffunc.read_tiff(output_name).astype('float')
#     blobs_list = CE.image_blobs(new_slice, blob_set=[5,3,10], th = 120., OL=1.)
#     print("Extracted cells:", len(blobs_list))
#     print(blobs_list)

if __name__ == '__main__':
    main()

