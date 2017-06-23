'''
Analyze temporal features of existing data.
Created by Dan on 06/21/2017
Last update:
'''

import sys
global_packagepath = '/home/sillycat/Programming/Python/Image_toolbox/'
sys.path.append(global_packagepath)
import numpy as np
from scipy import signal


def temporal_filter(dff_raw, tpfilter, dt = 0.50):
    '''
    section the parts withint tpfilter and return the filtered signal
    '''
    NT, NP = dff_raw.shape
