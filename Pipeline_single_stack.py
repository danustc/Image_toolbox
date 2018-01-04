'''
Extract cells and do the dff calculation of one image stack. Assume that the stack is drift-free.
'''
import sys
import h5py
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/src/')
import numpy as np
import shared_funcs.tifffunc as tf
import networks.df_f as dff
import preprocessing.cell_extract as cell_extract
import visualization.brain_navigation as brain_navigation
import visualization.signal_plot as signal_plot
import networks.simple_variance as simple_variance
import matplotlib.pyplot as plt
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

def main():
    data_path = global_datapath + 'Streaming_phasor.tif'
    str_stack = tf.read_tiff(data_path)
    messed_slices = np.array([8, 17, 25, 34, 43, 60])
    str_stack[messed_slices] = (str_stack[messed_slices-1]+str_stack[messed_slices+1])/2.0
    CE = cell_extract.Cell_extract(str_stack, diam = 8)
    cblobs = CE.extract_sampling([0,5,10,15, 20, 25, 30], mode = 'a', bg_sub = 45, red_reduct = 4)
    print(cblobs[:,:2])
    train_signal = CE.stack_signal_propagate(cblobs)
    dffr, f_base = dff.dff_raw(train_signal, ft_width = 5, ntruncate = 0)
    print(f_base.shape)
    dffr[dffr<-0.02] = -0.02
    hf = h5py.File('Streaming_extract.h5', 'w')
    hf.create_dataset('coords', data = cblobs)
    hf.create_dataset('signdl', data = train_signal)
    hf.close()
    coord_stimulate = [125, 123] # the coordinate of the stimulated neuron
    yx_coord = cblobs[:,:2]
    dist = yx_coord - coord_stimulate
    dist_rank = np.argsort(dist[:,0]**2 + dist[:,1]**2)
    print(yx_coord[dist_rank[0]])
    NT = dffr.shape[0]
    fig_stimu = plt.figure(figsize = (7,4))
    ax = fig_stimu.add_subplot(111)
    ax.plot(np.arange(NT)*0.60, dffr[:,dist_rank[0]])
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Delta F/F')
    fig_stimu.savefig(global_datapath + 'stimulated')
    plt.clf()
    plt.plot(np.arange(NT)*0.60, f_base[:,dist_rank[0]])
    plt.savefig(global_datapath + 'stimulated_base')
    fig_d = brain_navigation.slice_display(cblobs, title = 'Extracted neurons', ref_image = str_stack[0])
    fig_d.savefig(global_datapath + 'Streaming_extracted')

    crank = simple_variance.simvar_global_sort(dffr)[0]
    n_best = 15
    fig_n1 = signal_plot.dff_rasterplot(dffr[:,crank[:50]], dt = 0.8)
    fig_n1.savefig(global_datapath + 'Streaming_dff_1')

    fig_n2 = signal_plot.dff_rasterplot(dffr[:,crank[50:100]], dt = 0.8)
    fig_n2.savefig(global_datapath + 'Streaming_dff_2')

    fig_n3 = signal_plot.dff_rasterplot(dffr[:,crank[100:150]], dt = 0.8)
    fig_n3.savefig(global_datapath + 'Streaming_dff_3')

    fig_n4 = signal_plot.dff_rasterplot(dffr[:,crank[150:]], dt = 0.8)
    fig_n4.savefig(global_datapath + 'Streaming_dff_4')
if __name__=='__main__':
    main()

