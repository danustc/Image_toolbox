"""
Created by Dan on 08/15/2016, a test of data processing pipeline
Last update: 06/15/2017, some major changes are made.
Although the low-frequency background is subtracted, cell extraction is still performed on the uncorrected image. This may help eliminating artifacts.
This is the windows version. Don't mix it with the linux version!
"""
package_path_win ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
package_path_ubn='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'

import os
import sys
import glob
import numpy as np
sys.path.append(package_path_win)
import matplotlib.pyplot as plt
from PIL import Image
from src.preprocessing.segmentation import *
#from src.preprocessing.drift_correction import DC_pipeline

data_rootpath_win ='X:/Zebrafish_ispim/2018-09-24/Sep24_2018_B2/\\'
data_rootpath_yst ='Z:/Dan/Data_Rock/Sep24_2018_A2/\\'
data_rootpath_portable ='/media/sillycat/DanData/Jul19_2017_A2/\\'
#folder_list = glob.glob(data_rootpath+"/B3_TS\\")
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

    def load_file(self, nfile, size_th = 600, verbose = True):
        '''
        Load the nfileth data file.
        size_th: threshold of the data file. Unit: MB.
        '''
        fname = self.raw_list[nfile]; print(fname)
        self.pim = Image.open(fname) # open the image
        n_frames = self.pim.n_frames
        w,h = self.pim.size
        stack_size= w*h*n_frames/(600*1024)

        if(stack_size > size_th):
            self.stepload = True
            n_groups = int(stack_size/size_th)+1
            ss_slices = int(n_frames/n_groups)+1 # number of slices to import everytime
            stack_cut = np.zeros(n_groups + 1)
            stack_cut[:n_groups] = np.arange(n_frames, step = ss_slices)
            stack_cut[-1] = n_frames# the cutting-off positions
            self.stack_cut = stack_cut.astype('uint16')
            self.n_groups = n_groups
            if verbose:
                print("number of group:", n_groups)
                print("Cutting-off places:", stack_cut)
        else:
            self.stepload = False
        self.current_file = nfile


    def sampling(self, nsamples, verbose = True):
        '''
        Read nsamples from the filehandle
        nsamples is an array or a list
        '''
        sample_stack = []
        for ns in nsamples:
            self.pim.seek(ns)
            sample_stack.append(np.array(self.pim)) # this step is pretty time consuming)
        
        sample_stack = np.array(sample_stack)

        blobs_sample = stack_blobs(sample_stack, self.cdiam, sig = 5)
        self.cblobs = stack_redundreduct(blobs_sample, th = 5) # redundancy removed substack, saves the y,x coordinates of the extracted blobs
        if verbose:
            print("Done with sampling! Number of blobs:", self.cblobs.shape[0])

    def process(self, verbose = True):
        '''
        suppose the sampling has been done, and current file has been processed.
        If size_th > 500 GB, load stepwize
        '''
        cblobs = self.cblobs
        fname_stem = os.path.splitext(self.raw_list[self.current_file])[0]
        # stepload or one-time load?
        if(self.stepload):
            signal_series = [] # create an empty list, which should be merged later
            
            for n_step in range(3,self.n_groups):
                si = self.stack_cut[n_step]
                sf = self.stack_cut[n_step+1]
                substack = []
                print(si,sf)
                for ii in range(si, sf):
                    self.pim.seek(ii)
                    substack.append(np.array(self.pim))
    
                #print(substack[0].shape)
                substack = np.array(substack)
                #print(substack.shape)
                sub_time_series = stack_signal_propagate(substack, cblobs) # return
                signal_series.append(sub_time_series)
                if verbose:
                    print("Processed step ", n_step)
            # now, let's concatenate the substacks in the list and compile it into a new dataset 
            sample_frame = substack[0]
            ts_signal = np.concatenate(tuple(signal_series), axis = 0)

        else: # one-time load
            raw_stack = self.tif_handle.asarray()
            sample_frame = raw_stack[0]
            ts_signal = stack_signal_propagate(raw_stack, cblobs)

        self.ts_dataset = position_signal_compile(cblobs, ts_signal)
        np.savez(fname_stem, **self.ts_dataset)
        self.pim.close()
        self.process_log[self.current_file] = 3

        sample_frame = frame_deblur(sample_frame, sig = 5, Nit = 15)
        fig_display  = plt.figure(figsize = (8,5.6))
        ax = fig_display.add_subplot(111)
        ax.imshow(sample_frame, cmap = 'Greys_r')
        ax.scatter(cblobs[:,1], cblobs[:,0], s = 7, color = 'g')
        ax.axis('off')
        ax.set_title('# of blobs: '+ str(cblobs.shape[0]), fontsize = 16)
        fig_display.tight_layout()
        fig_display.savefig(fname_stem + '_cells')
        plt.close(fig_display)



    def run_pipeline(self, n_samples, verbose = True):
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
            self.process()

    # should I also define an exit function? 

# -----------------------The main test function -----------------------
def main():
    #folder_list = glob.glob(data_rootpath_yst+"/A2_TS/\\")
    folder_list = glob.glob(data_rootpath_win+"/B2_TS/\\")
    for data_path in folder_list:
        print(data_path)
        pt = pipeline_tstacks(data_path, fname_flags = 'rg')
        pt.run_pipeline([20,50,100, 200])
    #pt = pipeline_tstacks(data_path2, fname_flags = 'ZP')
    #pt.run_pipeline([5,10,15,20])


if __name__ == '__main__':
    main()
