import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

def stride_division(nl, n_pat, nstep, offset = 0):
    '''
    rearrange np.arange(nl) into small segments with np width and nstep stride size.
    '''
    arr = np.arange(nl - offset)
    nw = int((nl -offset - npat)/nstep)+1 # ok this is correct.
    patches_raw = np.tile(np.arange(n_pat) + offset, (nw,1)).T + np.arange(nw)*nstep
    patches = patches_raw.T
    return patches


def patch_finding_2d(img, patch_size = (512,512), stride = 20, offset = (0,0)):
    '''
    img: the raw image
    patch: the size of the patch
    stride: patch finding step
    '''
    off_y, off_x = offset
    NR, NC = img.size
    pr, pc = patch_size
    pat_r = stride_division(NR, pr, stride, off_y)
    pat_c = stride_division(NC, pc, stride, off_x)

    patches = []

    for sec_r in pat_r:
        for sec_c in pat_c:
            patches.append(img[sec_r, sec_c])

    return patches


def main():
    pass

if __name__ == '__main__':
    main()
