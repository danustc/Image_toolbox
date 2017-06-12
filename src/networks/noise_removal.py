'''
Removing high-frequency noises (those wrongly extracted cells)
'''

import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import numpy as np
import pyfftw

def high_freq_detect(dff_raw, fcut):
    '''
    dff_raw: the raw Delta F/F signals, each column represents DF/F of a cell.
    fcut: the cutting frequency
    return: the indices of the cells that show little activities.
    '''
