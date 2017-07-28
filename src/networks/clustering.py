'''
Created on 07/27/2017 by Dan. Clustering of the data.
Visualization is inherently included here.
Last modification:
'''
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def dis2cluster(dataset):
    Z = linkage(dataset, 'ward')
    figc = plt.figure()
    ax = figc.add_subplot(111)
    ax.set_ylabel('distance')
    dendrogram(Z, leaf_rotation = 90.)
    return figc
