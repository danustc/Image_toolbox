'''
Visualization tool of anatomical labeling statistics.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def label_scatter(mask_counts, mask_name, color = 'coral'):
    '''
    mask_counts: the count of each  
    '''

