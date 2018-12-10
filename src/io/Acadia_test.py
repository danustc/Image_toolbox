'''
This is a script for Acadia test
'''

global_datapath = '/home/sillycat/Programming/Python/data_test/Acadia/'
import numpy as np
from datareader import DaxReader
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from sklearn.metrics import pairwise_distances
import tifffile as tf

def psfstack_survey(stack, thresh = 150):
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
    blobs = blob_log(dominant_slice, min_sigma = min_sig, max_sigma = max_sig, threshold = 40 )
    centers = blobs[:,:2] # blob_centers
    lower_rest = np.logical_and(centers[:,0] > hps, centers[:,1] > hps)
    upper_rest = np.logical_and(centers[:,0] < NY - hps, centers[:,1] < NX-hps)
    total_rest = np.logical_and(lower_rest, upper_rest)
    centers = centers[total_rest]

    ind_accept = psf_isolation(centers, 30)
    centers = centers[ind_accept]
    print(centers)


    psf_collection = []
    for cc in centers:
        psf_patch = stack[:, cc[0]-hps:cc[0]+hps, cc[1]-hps:cc[1]+hps]
        psf_collection.append(psf_patch)

    return psf_collection, centers


def psf_isolation(coord_list, iso_thresh):
    '''
    Select coordinates that are well separated from one another.
    '''
    NC = coord_list.shape[0] # number of psfs
    dist_mat = pairwise_distances(coord_list, metric = 'euclidean')
    dist_mat = dist_mat + np.eye(NC)*256
    dmin = dist_mat.min(axis = 0)
    ind_accept = np.where(dmin > iso_thresh)[0] # even the min distance is above the threshold

    return ind_accept

def zcoef():
    fpath = '2018-11-14/Z_coef_1114_tune1.npy'
    zcoef =  np.load(global_datapath + fpath)
    zm = zcoef.mean(axis = 0)
    zs = zcoef.std(axis = 0)
    fig = plt.figure(figsize = (5,3))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(4,12)+1, zm[4:], yerr = zs[4:])
    ax.set_ylim([-0.03, 0.01])
    ax.tick_params(labelsize = 14)
    fig.savefig('bar_mod1')





def main():
    fpath = '2018-11-14/TL150_tune_0001.dax'
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
    tf.imsave('imstack.tif', stack.astype('uint16'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(stack[dom_z])
    psf_collection, centers = psf_finder(stack, dom_z)
    ax.scatter(centers[:,1], centers[:,0], s = 150, facecolors = 'none',edgecolors = 'yellow', linewidth = 2 )
    ax.axis('off')
    plt.tight_layout()
    fig.savefig('PSF_extract')
    print("# of psfs:",len(psf_collection))
    ii = 0
    for psf in psf_collection:
        print(centers[ii])
        ax.imshow(psf[:, 23, :], cmap = 'Greys_r', extent = [-0.088*24, 0.088*24, -3, 3])
        ax.axis('off')
        fig.savefig('psf_xz'+str(ii)+'tune1')
        ii+=1

if __name__ == '__main__':
    main()
