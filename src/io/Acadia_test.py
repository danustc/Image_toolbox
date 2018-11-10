'''
This is a script for Acadia test
'''

global_datapath = '/home/sillycat/Programming/Python/data_test/Acadia/'
import numpy as np
from datareader import DaxReader
import matplotlib.pyplot as plt
from skimage.feature import blob_log


def psfstack_survey(stack, thresh = 1500):
    '''
    evaluate the number of PSFs in the image stack.
    thresh: pixel intensity threshold
    '''
    ind_zmax = np.argmax(stack, axis = 0)
    zmax = np.max(stack, axis = 0)

    zy, zx = np.where(zmax > thresh)
    zz = ind_zmax[zy,zx]
    uz = np.unique(zz) # unique zz
    hist, be = np.histogram(zz, bins = uz)
    dominant_z = be[hist.argmax()]

    return zz, dominant_z

def psf_finder(stack, dominant_z, px = 100, wl = 700, patch_size = 48):
    '''
    find psfs and their centers from a stack.
    patch_size: the size of each psf small stack.
    '''
    hps = int(patch_size //2)
    NY, NX = stack[0].shape
    dominant_slice = stack[dominant_z]
    min_sig = 0.61*wl/px #theoretical diffraction-limited psf size at focus
    max_sig = 1.5* min_sig # this is kinda random
    blobs = blob_log(dominant_slice, min_sigma = min_sig, max_sigma = max_sig, threshold = 100 )
    centers = blobs[:,:2] # blob_centers
    print(centers)
    lower_rest = np.logical_and(centers[:,0] > hps, centers[:,1] > hps)
    upper_rest = np.logical_and(centers[:,0] < NY - hps, centers[:,1] < NX-hps)
    total_rest = np.logical_and(lower_rest, upper_rest)
    centers = centers[total_rest]

    psf_collection = []
    for cc in centers:
        psf_patch = stack[:, cc[0]-hps:cc[0]+hps, cc[1]-hps:cc[1]+hps]
        psf_collection.append(psf_patch)

    return psf_collection


def psf_isolation(coord_list, iso_thresh):
    '''
    Select coordinates that are well separated from one another.
    '''





def main():
    fpath = '2018-11-02/RL150_0001.dax'
    DM = DaxReader(global_datapath + fpath)
    print(DM.image_height)
    print(DM.image_width)
    print(DM.number_frames)
    stack = []
    for cc in range(DM.number_frames):
        frame = DM.loadAFrame(cc)
        stack.append(frame.astype('float64'))

    stack = np.array(stack)
    zz, dom_z = psfstack_survey(stack)

    psf_collection = psf_finder(stack, dom_z)
    print("# of psfs:",len(psf_collection))
    plt.imshow(psf_collection[0][dom_z])
    plt.show()

if __name__ == '__main__':
    main()
