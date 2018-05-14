'''
Extract cells and do the dff calculation of one image stack. Assume that the stack is drift-free.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/src/')
import numpy as np
import shared_funcs.tifffunc as tf
import networks.df_f as dff
import preprocessing.segmentation as segmentation
import visualization.brain_navigation as brain_navigation
import visualization.signal_plot as signal_plot
import networks.simple_variance as simple_variance
import matplotlib.pyplot as plt
global_datapath = '/home/sillycat/Programming/Python/data_test/Light_stimulus/'

def main():
    data_path = global_datapath + '0410_Z3.tif'
    str_stack = tf.read_tiff(data_path).astype('float64')
    #messed_slices = np.array([8, 17, 25, 34, 43, 60])
    #str_stack[messed_slices] = (str_stack[messed_slices-1]+str_stack[messed_slices+1])/2.0
    CE = segmentation.Cell_extract(str_stack, diam = 5)
    cblobs = CE.extract_sampling(np.arange(15), mode = 'a', bg_sub = -1, red_reduct = 4)
    print(cblobs[:,:2])
    train_signal = CE.stack_signal_propagate(cblobs)

    raw_data = dict()
    coords = np.zeros([len(cblobs),3])
    raw_data['coord'] = cblobs
    raw_data['data'] = train_signal
    np.savez(global_datapath+'0410_Z3.npz', **raw_data)
    print(cblobs.shape)


    # display the locations
    fig = brain_navigation.slice_display(slice_blobs = cblobs[:,:2], ref_image = str_stack[:15].mean(axis = 0), title = '0418_Z3')
    fig.savefig(global_datapath+'Z3_Cells.png')



if __name__=='__main__':
    main()

