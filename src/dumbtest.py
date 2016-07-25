from Preprocess import Preprocess
import matplotlib.pyplot as plt
import tifffunc
import trackpy as tp
import numpy as np
from skimage.filters import threshold_adaptive
from skimage import color, io, filters


def main():
#     datapath = 'C:\username\Documents\......(fill it up)'
    input_name = 'test_image' # replace this with your image name.
    sig = 30 # play with this number 
    impp = Preprocess(input_name, sig)
    new_stack = impp.stack_high_trunc()
    output_name = input_name+str(sig)
    tifffunc.write_tiff(new_stack, output_name)


if __name__ == '__main__':
    main()

