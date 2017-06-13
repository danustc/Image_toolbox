"""
A wrapper designed for Dan's image processing.
Based on Christoph Gohlke (UCI)'s tifffile module.
Last modification: 06/02/17
"""
import os
from tifffile import TiffFile, imsave
import numpy as np


# read a tiff stack
def tiff_describe(fname, handle_open = True):
    '''
    describe the properties of the tiff stack: number of slices, y,x pixels
    '''
    with TiffFile(fname) as tif:
        stack_size = tif.fstat.st_size/(1024*1024)
        stack_shape = tif.series[0].shape # imageJ description of the stack, in bytes instead of str

    if handle_open:
        return stack_size, stack_shape, tif
    else:
        tif.close() # remember to close the tif to save the memory.
        return stack_size,stack_shape


def read_tiff(fname, nslice = None):
    # the fname should include the absolute path and extension name
    # nslice can be a number or an array, indicating multiple slices
    with TiffFile(fname) as tif:
        istack = tif.asarray()
    if nslice is None:
        return np.copy(istack)
    else:
        return np.copy(istack[nslice])
    tif.close()


def write_tiff(imstack, fname):
    # assume that fname already has the extension.
    imsave(fname, imstack.astype('uint16'))



def crop_tiff(imstack,positions, cfname):
    '''
    crop a tiff image
    imstack is already an np array
    '''
    yi = positions[0]
    yf = positions[2]+ yi
    xi = positions[1]
    xf = positions[3]+ xi
    cr_stack = imstack[:,yi:yf, xi:xf]
    return cr_stack
