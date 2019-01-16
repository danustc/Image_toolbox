"""
Created by Dan on 08/15/2016, a test of data processing pipeline
Last update: 12/24/2018, add "fake cells" into the dataset.
Although the low-frequency background is subtracted, cell extraction is still performed on the uncorrected image. This may help eliminating artifacts.
This is the windows version. Don't mix it with the linux version!
"""
import os
import sys
import glob
import time
import numpy as np
from PIL import Image as pilimage
import matplotlib.pyplot as plt
from itbx.preprocessing.segmentation import *
import itbx.preprocessing.background_estimation as back_es

data_rootpath_win ='D:/Data/2018-08-02/Aug02_2018_B5\\'
data_rootpath_yst ='D:/Dan/Data_Rock/Sep24_2018_B5\\'
data_rootpath_portable ='/media/sillycat/DanData/'
#folder_list = glob.glob(data_rootpath+"/B5_TS\\")
# -----------------------------------------Big classes-------------------------------------------------

class pipeline_tstacks(object):
    '''
    Goal: process all the t-stacks in a folder
    0. Initialize: load the folder and load the files
    1. extract cells and calculate the signals in each t-stack. If t-stack is too large, do this in steps.
    2. Concatenate the processed key-values in a large dataset, then save.
    3. Report progress.
    '''

    def __init__(self, work_folder, fname_flags = '_ZP_', cdiam = 9):
        '''
        work_folder: the folder that contains all the .tif files
        '''
        self.work_folder = work_folder
        print(work_folder)
        self.raw_list = glob.glob(work_folder + '*'+fname_flags + '*.tif')
        self.current_file = None # which data set am I working on?
        self.cdiam = cdiam

        Nfile = len(self.raw_list)
        if (Nfile ==0):
            print("Error! The folder does not contain any files.")
        else:
            self.Nfile = Nfile
            print("There are", Nfile, "image stacks to be processed.")
            self.process_log= np.zeros(Nfile)
            '''
            process_log:    0 --- unprocessed
                            1 --- loaded
                            2 --- sampled
                            3 --- extracted
                            -1 --- error
            '''
        return

    def load_file(self, nfile, size_th = 300, verbose = True):
        '''
        Load the nfileth data file.
        size_th: threshold of the data file. Unit: MB.
        '''
        fname = self.raw_list[nfile]; print(fname)
        self.im = pilimage.open(fname)
        w, h = self.im.size
        nslices = self.im.n_frames
        stack_size = 2*w*h*nslices/(1024*1024)


        if(stack_size > size_th):
            self.stepload = True
            n_groups = int(stack_size/size_th)+1
            ss_slices = int(nslices/n_groups)+1 # number of slices to import everytime
            stack_cut = np.zeros(n_groups + 1)
            stack_cut[:n_groups] = np.arange(nslices, step = ss_slices)
            stack_cut[-1] = nslices # the cutting-off positions
            self.stack_cut = stack_cut.astype('int')
            self.n_groups = n_groups
            if verbose:
                print("number of group:", n_groups)
                print("Cutting-off places:", stack_cut)
        else:
            self.stepload = False

        self.current_file = nfile
        self.nslices = nslices


    def sampling(self, nsamples, verbose = True):
        '''
        Read nsamples from the filehandle
        nsamples is an array or a list
        '''
        sample_stack = []
        for ns in nsamples:
            self.im.seek(ns)
            sample_stack.append(np.array(self.im))
        time.sleep(1)
        sample_stack = np.array(sample_stack)

        #sample_stack = self.tif_handle.asarray()[nsamples] # this step is pretty time consuming

        blobs_sample = stack_blobs(sample_stack, self.cdiam, sig = 5)
        self.cblobs = stack_redundreduct(blobs_sample, th = 5) # redundancy removed substack, saves the y,x coordinates of the extracted blobs
        if verbose:
            print("Done with sampling! Number of blobs:", self.cblobs.shape[0])

    def process(self, verbose = True, add_fake = True):
        '''
        suppose the sampling has been done, and current file has been processed.
        after the sampling and extraction of blobs has been done, calculate F values in each blob through time.
        If size_th > 500 GB, load stepwize
        '''
        cblobs = self.cblobs
        fname_stem = os.path.splitext(self.raw_list[self.current_file])[0]
        if add_fake:
            fake_series = [] # create an empty list, which should be merged later
            frame = np.array(self.im)
            PR, PC = back_es.binning_cutoffs(frame.shape, grid_size = 10)
            pcbins = back_es.frame_binning(frame, PR, PC)
            vb = back_es.background_found(self.im, nslice = 3)
            print(vb)
            pc_range = np.logical_and(pcbins > vb[0], pcbins < vb[1])
            vr, vc = np.where(pc_range)
            cr, cc = back_es._voxel_recover_(vr, vc, grid_size = 10)
            fblobs = np.c_[cr, cc]
            fblobs = fblobs[::5] # added by Dan
            print(fblobs.shape)
        # stepload or one-time load?
        if(self.stepload):
            signal_series = [] # create an empty list, which should be merged later
            si = self.stack_cut[0]
            for n_step in range(self.n_groups):
                sf = self.stack_cut[n_step+1]
                substack = []
                for ii in range(si, sf):
                    self.im.seek(ii)
                    substack.append(np.array(self.im))

                time.sleep(2)
                substack = np.array(substack)
                sub_time_series = stack_reextract(substack, cblobs) # return
                signal_series.append(sub_time_series)
                if add_fake:
                    sub_fake_series = stack_reextract(substack, fblobs, dr = 5)
                    fake_series.append(sub_fake_series)
                si = sf

                if verbose:
                    print("Processed step ", n_step)
            # now, let's concatenate the substacks in the list and compile it into a new dataset 
            sample_frame = substack[0]
            ts_signal = np.concatenate(tuple(signal_series), axis = 0)
            if add_fake:
                ts_fake = np.concatenate(tuple(fake_series), axis = 0)

        else: # one-time load
            raw_stack = []
            for ii in range(self.nslices):
                self.im.seek(ii)
                raw_stack.append(np.array(self.im))

            raw_stack = np.array(raw_stack)
            #sample_frame = raw_stack[0]
            ts_signal = stack_reextract(raw_stack, cblobs)

        self.ts_dataset = position_signal_compile(cblobs, ts_signal)
        np.savez(fname_stem, **self.ts_dataset)
        if add_fake:
            self.fk_dataset = position_signal_compile(fblobs, ts_fake)
            np.savez(fname_stem+'_fk', **self.fk_dataset)
        self.im.close()
        self.process_log[self.current_file] = 3

        sample_frame = frame_deblur(sample_frame, sig = 5, Nit = 21)
        fig_display  = plt.figure(figsize = (8,5.6))
        ax = fig_display.add_subplot(111)
        ax.imshow(sample_frame, cmap = 'Greys_r')
        ax.scatter(fblobs[:,1], fblobs[:,0], s = 7, color = 'g')
        ax.axis('off')
        ax.set_title('# of blobs: '+ str(fblobs.shape[0]), fontsize = 16)
        fig_display.tight_layout()
        fig_display.savefig(fname_stem + '_cells')
        plt.close(fig_display)



    def run_pipeline(self, n_samples, verbose = True, add_fake = True):
        '''
        run the pipeline
        '''
        for zp in range(self.Nfile):
            '''
            0 --- load file
            1 --- Sampling
            2 --- cell extraction and save
            '''
            self.load_file(zp)
            if verbose:
                print("Current file:", self.current_file)
            self.sampling(n_samples)
            self.process(verbose, add_fake)


    # should I also define an exit function? 

# -----------------------The main test function -----------------------
def main():
    folder_list = glob.glob(data_rootpath_portable+"Jul*2017*/*TS/")


    #folder_list = glob.glob(data_rootpath_win+"/B2_TS/\\")
    for data_path in folder_list:
        print(data_path)
        pt = pipeline_tstacks(data_path, fname_flags = 'rg')
        pt.run_pipeline([50,100,200,400])
    #pt = pipeline_tstacks(data_path2, fname_flags = 'ZP')
    #pt.run_pipeline([5,10,15,20])


if __name__ == '__main__':
    main()
