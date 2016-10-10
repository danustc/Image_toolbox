"""
Recreated by Dan on 10/10/2016. 
Test redundancy removal.
"""


import os 
from Postprocess import brain_construct
import numpy as np
import matplotlib.pyplot as plt 
import tifffunc

def dumb1():
    path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/A1_TS_rest/'
    
    BC = brain_construct(path, dims = [800,760])
    new_stack = BC.zs_construct()
    tifffunc.write_tiff(new_stack, path+'new.tif')
    
    
#     cmap = BC.corrmap_stack()
#     dmap = BC.dist_map
    
    

if __name__ == '__main__':
    dumb1()
