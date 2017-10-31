'''
Extract cells and do the dff calculation of one image stack. Assume that the stack is drift-free.
'''
import sys
import h5py
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/src/')
import numpy as np
import shared_funcs.tifffunc as tf
import preprocessing.cell_extract as cell_extract
import visualization.brain_navigation as brain_navigation

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

def main():
    data_path = global_datapath + 'Streaming_phasor.tif'
    str_stack = tf.read_tiff(data_path)
    CE = cell_extract.Cell_extract(str_stack, diam = 8)
    cblobs = CE.extract_sampling([0,5,10,15, 20], mode = 'a', bg_sub = 30, red_reduct = 4)
    print(cblobs.shape)
    train_signal = CE.stack_signal_propagate(cblobs)
    print(train_signal.shape)



if __name__=='__main__':
    main()

