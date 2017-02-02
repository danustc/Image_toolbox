"""
A wrapper designed for Dan's image processing.
Based on Christoph Gohlke (UCI)'s tifffile module.
Last modification: 08/15/16
"""

from tifffile import TiffFile
from tifffile import imsave
import numpy as np

# read a tiff stack
def read_tiff(fname):
    # the fname should include the absolute path and extension name
    with TiffFile(fname) as tif:
        istack = tif.asarray()
    return istack

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



def main():
    impath = '/home/sillycat/Documents/Zebrafish/Exp_figures/ZD_2.tif'
    tiff = read_tiff(impath)
    reorient_tiff_RAS(tiff, impath[:-4]+ '_rio.tif')

if __name__ == '__main__':
    main()
