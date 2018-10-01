'''
Visualization tool of anatomical labeling statistics.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def label_scatter(mask_counts, color = 'coral'):
    '''
    mask_counts: the count of each mask
    '''
    ax = sns.boxplot(data = mask_counts, orient = 'h')
    #ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

    return ax


#label_scatter()


