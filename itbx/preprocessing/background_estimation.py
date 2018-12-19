import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import argrelextrema
from correlation import fft_image

def binning_cutoffs(frame_size, n_frames = 3, grid_size = 10):
    '''
    bin an image into frames
    '''
    # Q: Is it necessary? 
    nr, nc = frame_size
    MR = int(nr//grid_size)
    MC = int(nc//grid_size)

    patch_R = np.arange(MR)*grid_size
    patch_C = np.arange(MC)*grid_size

    return patch_R, patch_C


def read_substack(pims, keys):
    '''
    read a range of frames
    '''
    substack = []
    for ff in range(keys[0], keys[1]):
        pims.seek(ff)
        substack.append(np.array(pims))

    substack = np.array(substack)

    return substack



def frame_binning(frame, PR, PC):
    MR = len(PR)-1
    gs = PR[1] - PR[0]
    MC = len(PC)-1
    pc = np.zeros((MR, MC))
    for ii in range(MR):
        for jj in range(MC):
            pc[ii, jj]= frame[PR[ii]:PR[ii]+gs,  PC[jj]:PC[jj]+gs].mean()

    return pc

def stack_binning(stack, PR, PC):
    NF = stack.shape[0]
    MR = len(PR)-1
    gs = PR[1] - PR[0]
    MC = len(PC)-1
    pc = np.zeros((NF, MR, MC))
    for ii in range(MR):
        for jj in range(MC):
            pc[:, ii, jj]= stack[:, PR[ii]:PR[ii]+gs,  PC[jj]:PC[jj]+gs].mean(axis = (1,2))

    return pc



def rawf_vox(pims, PR, PC, n_chop = 5):
    '''
    pims: a pointer to the Image sequence
    PR, PC: the cutoff positions
    '''
    NF = pims.n_frames
    chop_position = np.linspace(0, NF, n_chop + 1).astype('uint16')
    rawf_stack = []

    for ff in range(n_chop):
        substack = read_substack(pims, (chop_position[ff], chop_position[ff+1]))
        print(ff) # this is fucking slow.
        fpc = stack_binning(substack, PR, PC)
        rawf_stack.append(fpc)

    rawf_stack = np.array(rawf_stack)
    return rawf_stack


def _diff_peaks_(hist):
    '''
    detect peaks in a histogram
    '''
    peaks = argrelextrema(hist, np.greater_equal, order = 2)
    valleys = argrelextrema(hist, np.less_equal, order = 2)
    print(peaks)
    print(valleys)
    return peaks[0], valleys[0]



def background_found(pim, grid_size, nslice = 3):
    '''
    input: a PIL.Image handle, grid size (unit of pixels)
    '''
    sub_stack = []
    for ii in range(nslice):
        pim.seek(ii)
        sub_stack.append(np.array(pim))

    sub_stack = np.array(sub_stack)
    mrange = np.max(sub_stack.flatten()*0.6)

    hist, be = np.histogram(sub_stack, bins = 100, range = (100, mrange)) # create a intensity distribution of histogram
    plt.bar(be[:-1], hist, width = 15)
    plt.show()
    pp, vv = _diff_peaks_(hist)
    ivp = np.searchsorted(pp, vv)
    print(ivp)
    #v_background = vv[ivp ==1] # find the first valley
    v_background = pp[0] # find the first valley
    print("background level:", be[v_background], be[v_background+1])
    mh = np.argmax(hist)
    int_peak = (be[mh] + be[mh+1])*0.5
    print(int_peak)

    return be[v_background:v_background+2]



def main():
    data_path = '/media/sillycat/DanData/Jul19_2017_A2/A2_TS/'
    plist = glob.glob(data_path + 'rg*.tif')
    pname = plist[10]
    print(pname)
    pim = Image.open(pname)
    frame = np.array(pim)
    PR, PC = binning_cutoffs(frame.shape)
    pcbins = frame_binning(frame,PR, PC)

    print(frame.shape)
    vb = background_found(pim, 10, nslice = 5)
    print(vb)
    pc_range = np.logical_and(pcbins > vb[0], pcbins < vb[1])
    rr, cc = np.where(pc_range)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(pcbins, cmap = 'Greys_r')
    ax1.scatter(cc,rr, color = 'r', s = 1)
    ax2.imshow(pc_range, cmap = 'Greens_r')
    plt.show()
    plt.close()

    ft_frame = fft_image(frame)
    plt.imshow(np.log(ft_frame))
    plt.show()
    pim.close()

if __name__ == '__main__':
    main()
