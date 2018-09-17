package_path = '/home/sillycat/Programming/Python/Image_toolbox/'

import numpy as np
import tifffile as tf
import sys
sys.path.append(package_path)
import src.preprocessing.segmentation as segmentation
import glob
import matplotlib.pyplot as plt

data_path_ptb = '/media/sillycat/DanData/Test_pool/'



def main():
    flist = glob.glob(data_path_ptb+'*.tif')
    stack = tf.imread(flist)
    print(stack.shape)
    fig_comp = plt.figure(figsize = (12,5))
    db_stack = []
    db_blobs = []
    for frame in stack:
        db_frame = segmentation.frame_deblur(frame, sig = 4, wd = 8, Nit = 20, mode = 'gauss' )
        db_stack.append(db_frame)
        cblobs = segmentation.frame_blobs(db_frame)
        db_blobs.append(cblobs)


    ax1 = fig_comp.add_subplot(121)
    ax1.imshow(db_stack[0], cmap = 'Greys_r')
    ax1.scatter(db_blobs[1][:,1], db_blobs[1][:,0], s = 8)
    ax1.scatter(db_blobs[0][:,1], db_blobs[0][:,0], s = 8)


    ax2 = fig_comp.add_subplot(122)
    ax2.imshow(db_stack[1], cmap = 'Greys_r')
    ax2.scatter(db_blobs[0][:,1], db_blobs[0][:,0], s = 8)
    ax2.scatter(db_blobs[1][:,1], db_blobs[1][:,0], s = 8)

    plt.show()
if __name__ == '__main__':
    main()
