"""
A wrapper designed for Dan's image processing.
Based on Christoph Gohlke (UCI)'s tifffile module.
Last modification: 06/02/17
"""

from tifffile import TiffFile, imsave
import numpy as np


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
