"""
A wrapper designed for Dan's image processing.
Based on Christoph Gohlke (UCI)'s tifffile module.
Last modification: 04/04/17
"""

from tifffile import TiffFile, imsave
import numpy as np
import glob
import os


# read a tiff stack
def read_tiff(fname, nslice = None):
    # the fname should include the absolute path and extension name
    # nslice can be a number or an array, indicating multiple slices
    with TiffFile(fname) as tif:
        istack = tif.asarray()
    if nslice is None:
        return np.copy(istack)
    else:
        return np.copy(istack[nslice])

def intp_tiff(istack, ns1, ns2, nint = 1):
    # linear interpolation of slices between
    int_stack = np.zeros(shape = (nint,)+ istack.shape[1:])
    for ii in np.arange(nint + 2):
        alpha = ii/(nint + 1.)
        int_stack[ii] = istack[ns1]*(1-alpha) + istack[ns2]*alpha
    return int_stack.astype('uint16')  # return as unint16, tiff

def write_tiff(imstack, fname):
    # assume that fname already has the extension.
    imsave(fname, imstack.astype('uint16'))




def main():
    impath = '/home/sillycat/Programming/Python/Image_toolbox/cmtkRegistration/'

if __name__ == '__main__':
    main()
