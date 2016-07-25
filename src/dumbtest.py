from Preprocess import Preprocess
import matplotlib.pyplot as plt
import tifffunc
import trackpy as tp
import numpy as np
from skimage.filters import threshold_adaptive
from skimage import color, io, filters


def main():
    # remove the comment from the line below: 
#     datapath = 'C:\username\Documents\......(fill it up)'

 
    input_name = 'test_image' # replace this with your image name.
    sig = 30 # play with this number. Hint: create a for loop like:
    wt = 0.40 # play with this number as well  
#     for sig in np.arange(10, 40, step = 5):
    impp = Preprocess(input_name, sig)
    new_stack = impp.stack_high_trunc(wt)
    output_name = input_name+str(sig)
    tifffunc.write_tiff(new_stack, output_name)


    # next, let's add some cell extraction procedures.



if __name__ == '__main__':
    main()

