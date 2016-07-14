from tifffile import TiffFile
from tifffile import imsave
import numpy as np

# read a tiff stack
def read_tiff(fname):
    # the fname should include the absolute path.
    with TiffFile(fname + '.tif') as tif:
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
    imsave(fname+'.tif', imstack)
    
