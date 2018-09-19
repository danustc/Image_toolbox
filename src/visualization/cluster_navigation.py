import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

color_code = ['Greens', 'Greys', 'Purples', 'Reds', 'Blues']

def clusters_show(cluster_coords, vol_img = None, cm_list = None):
    '''
    cluster_coords: list, each element stores coordinates of cells in a cluster, saved in x-y order.
    '''
    n_clu = len(cluster_coords)
    fig = plt.figure(figsize = (8,6))

    ax = fig.add_subplot(111)
    if vol_img is None:
        pass
    else:
        ax.imshow(vol_img, cmap = 'Greys_r')

    for ii in range(n_clu):
        coord = cluster_coords[ii]
        ax.scatter(coord[:,0], coord[:,1], s = 9, c = coord[:,2], cmap = color_code) # put the Z-information into d

    ax.axis('off')
    return fig



