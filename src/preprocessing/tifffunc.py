"""
A wrapper designed for Dan's image processing.
Based on Christoph Gohlke (UCI)'s tifffile module.
Last modification: 04/04/17
"""

from tifffile import TiffFile
from tifffile import imsave
import numpy as np
import glob


# read a tiff stack
def read_tiff(fname, nslice = None):
    # the fname should include the absolute path and extension name
    # nslice can be a number or an array, indicating multiple slices
    with TiffFile(fname) as tif:
        istack = tif.asarray()
    if nslice is None:
        return istack
    else:
        return istack[nslice]

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

def reorient_tiff_RAS(imstack, fname):
    '''
    Reoriente the image stack in the RAS coordinate system.
    '''
    # nz, nx, ny = imstack.shape
    rot_stack = []
    for zslice in imstack:
        rot_stack.append(np.rot90(zslice, k=3))

    rot_stack = np.array(rot_stack)
    write_tiff(rot_stack, fname)


def interlacing_split(stack_ref, stack_aln, folder_path, name_flag):
    '''
    interlacing two stacks and split them into several substacks.
    '''



def main():
    impath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    ZD_name= 'ZD_2.tif'
    TS_name= 'TS_2.tif'
    ZD_stack = read_tiff(impath+ZD_name)
    TS_stack = read_tiff(impath+TS_name)



if __name__ == '__main__':
    main()
