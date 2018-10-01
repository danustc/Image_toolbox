'''
Visualization tool of anatomical labeling statistics.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def label_scatter(mask_counts, mask_name, color = 'coral'):
    '''
    mask_counts: the count of each mask
    '''
    ax = scatterplot(x = "Masks", y = "Counts", data = mask_counts)

    return ax


#label_scatter()


