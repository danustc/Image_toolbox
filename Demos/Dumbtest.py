"""
Recreated by Dan on 10/10/2016. 
Test redundancy removal.
"""


import os 
from Postprocess import brain_construct
from z_dense import z_dense_ref, z_dense_construct
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import tifffunc

def dumb1():
    zd_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/ZD_stacks/A2_ZD.npz'
    
#     BC = brain_construct(path, dims = [800,760])
#     new_stack = BC.zs_construct()
#     tifffunc.write_tiff(new_stack, path+'new.tif')
    z_dense = z_dense_construct(zd_path)
    
    
    zd_ref = z_dense_ref(z_dense, dims = [700,760])
    dist_3d = zd_ref.stack_red_detect()
    print(dist_3d.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dist_3d[:,0], dist_3d[:,1], dist_3d[:,2])

    plt.show()
    #     cmap = BC.corrmap_stack()
#     dmap = BC.dist_map
    
    

if __name__ == '__main__':
    dumb1()
