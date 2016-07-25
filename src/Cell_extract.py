import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.feature import blob_log
from skimage.morphology import reconstruction


def image_dilation_trunc(im0):
    image = filters.gaussian(im0.astype('float'),1.0)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method= 'dilation')
    return dilated

    
def image_blobs(im0, blob_set = [5,4,6], th = 50, OL = 1):
    # edited on 07/25/26 
    # added the adjacent 
    
    mx_sig = blob_set[0]
    mi_sig = blob_set[1]
    nsig = blob_set[2]
    blobs_log = blob_log(im0, max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL)
    
    # let's plot the images
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(im0, cmap='Greys_r')
    ax2 = fig.add_subplot(122)
    ax2.imshow(im0, cmap='Greys_r')
    
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), np.sqrt(2)*r, color='g', linewidth=1, fill=True)
        ax2.add_patch(c)
        #         print(r)
    plt.show()
    return blobs_log # The list of traced blobs 
    
    
def stack_blobs(stack, blob_set = [5,4,6], th = 50, OL = 1):
    # extract all the blobs 
    # This is not done 
    # Updated on 07/25/2016
    blobs_stack = {} # change into a dictionary 
    for nslice in np.arange(stack.shape[0]):
        blobs_log = image_blobs(stack[nslice], blob_set, th, OL)
        blobs_stack[nslice] = blobs_log
        
        
    return blobs_stack