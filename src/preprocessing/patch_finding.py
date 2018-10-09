'''
This is a small module for finding patches in a 2d array (i.e., an image)
'''
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

def stride_division(nl, n_pat, nstep, offset = 0):
    '''
    rearrange np.arange(nl) into small segments with np width and nstep stride size.
    '''
    arr = np.arange(nl - offset)
    nw = int((nl -offset - n_pat)/nstep)+1 # ok this is correct.
    patches_raw = np.tile(np.arange(n_pat) + offset, (nw,1)).T + np.arange(nw)*nstep
    patches = patches_raw.T
    return patches


def patch_finding_2d(img, patch_size = (512,512), stride = 20, offset = (0,0), return_position = True):
    '''
    img: the raw image
    patch: the size of the patch
    stride: patch finding step
    '''
    off_y, off_x = offset
    NR, NC = img.shape
    pr, pc = patch_size
    pat_r = stride_division(NR, pr, stride, off_y)
    pat_c = stride_division(NC, pc, stride, off_x)

    patches = []
    if return_position:
        r_start = pat_r[:,0]
        c_start = pat_c[:,0]
    else:
        r_start, c_start = None, None


    for sec_r in pat_r:
        for sec_c in pat_c:
            patches.append(img[sec_r][:,sec_c])

    return np.array(patches), r_start, c_start


def patch_opt(img, patch_size = (512, 512), stride = 50, offset = (20,20), mode = 'var'):
    '''
    find the patch with the maximum variance.
    if mode == 'mean':
        return the patch with maximum mean
    elif
    '''
    patches, r_start, c_start = patch_finding_2d(img, patch_size, stride, offset, True) # return patches and their starting positions
    print(patches.shape)
    if mode == 'var':
        pvar = np.var(patches, axis = (1,2)) # calculate each patch's variance
        pind = np.argmax(pvar)
    elif mode == 'mean':
        pmean = np.mean(patches, axis = (1,2))
        pind = np.argmax(pmean)
    else:
        pmv = np.mean(patches, axis = (1,2))*np.var(patches, axis = (1,2))
        pind = np.argmax(pmv)


    [CC,RR] = np.meshgrid(c_start, r_start)
    patch = patches[pind]
    rs, cs = RR.ravel()[pind], CC.ravel()[pind]
    return patch, [rs, cs]


def main():
    pass

if __name__ == '__main__':
    main()
