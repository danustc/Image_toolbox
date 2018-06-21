"""
Created by Dan on 08/15/2016, a test of data processing pipeline
Last update: 06/15/2017, some major changes are made.
Although the low-frequency background is subtracted, cell extraction is still performed on the uncorrected image. This may help eliminating artifacts.
This is the windows version. Don't mix it with the linux version!
"""
package_path ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'

import os
import sys
import glob
import numpy as np
sys.path.append(package_path)
import src.shared_funcs.tifffunc as tifffunc

from src.preprocessing.segmentation import *

# -----------------------------------------Big classes-------------------------------------------------

class pipeline_zstacks(object):
    """
    This pipeline processes all the zstacks inside a folder.
    Procedure:
    0. glob --- find all the z-stacks in the working folder.  (completed)
    1. load a stack of tif file (in z stack) (completed)
    2. Split the file name to get time-point number, convert the original integers to 3-digit integers (completed)
    2. Perform in-plane deblur on the z-stack, return a cleaner one (completed)
    3. Perform Cell extraction on the deblurred z-stack, save as 'npz' (completed)
    4. reconstruct t-stacks from z-stacks (can be run from outside)
    """
    def __init__(self, work_folder, fname_flags= 'ZD', cdiam = 9):
        '''
        Initialize the pipeline
        '''
        self.work_folder = work_folder
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
            self.CE_dpt = Cell_extract(im_stack = None, diam = cdiam) # the default diameter of the blob: 9

    def load_file(self, nfile, verbose = True):
        '''
        Load the nfileth data file.
        Since the Z stacks is usually small, the stepwise loading is dropped here.
        size_th: threshold of the data file. Unit: MB.
        '''
        fname = self.raw_list[nfile]
        stack_size, self.stack_shape, self.tif_handle = tifffunc.tiff_describe(fname, handle_open = True)
        if verbose:
            print('Loaded file:', self.raw_list[nfile])
            print('Stack shape:', self.stack_shape)
        self.current_file = nfile


    def process(self, verbose = True):
        '''
        suppose the sampling has been done, and current file has been processed.
        If size_th > 500 GB, load stepwize
        '''
        fname_stem = os.path.splitext(self.raw_list[self.current_file])[0]

        raw_stack = self.tif_handle.asarray()
        self.CE_dpt.stack_reload(raw_stack, refill = True)
        self.CE_dpt.stack_blobs(bg_sub = 40, verbose = True)

        self.tif_handle.filehandle.close()
        self.tif_handle.close() # close the tif handle 
        self.tif_handle = None
        self.process_log[self.current_file] = 3
        self.CE_dpt.save_data_list(fname_stem)

        if verbose:
            print("Done with file:", fname_stem)
        return


    def run_pipeline(self, verbose = True):
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
            self.process()
#--------------------------------the counterpart for tstacks--------------------------------

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
            self.CE_dpt = Cell_extract(im_stack = None, diam = cdiam) # the default diameter of the blob: 9
        return

    def load_file(self, nfile, size_th = 600, verbose = True):
        '''
        Load the nfileth data file.
        size_th: threshold of the data file. Unit: MB.
        '''
        fname = self.raw_list[nfile]
        stack_size, stack_shape, self.tif_handle = tifffunc.tiff_describe(fname, handle_open = True)
        if(stack_size > size_th):
            self.stepload = True
            n_groups = int(stack_size/size_th)+1
            nslices = stack_shape[0]
            ss_slices = int(nslices/n_groups)+1 # number of slices to import everytime
            stack_cut = np.zeros(n_groups + 1)
            stack_cut[:n_groups] = np.arange(nslices, step = ss_slices)
            stack_cut[-1] = nslices # the cutting-off positions
            self.stack_cut = stack_cut
            self.n_groups = n_groups
            if verbose:
                print("number of group:", n_groups)
                print("Cutting-off places:", stack_cut)
        else:
            self.stepload = False


        self.stack_shape = stack_shape
        self.current_file = nfile


    def sampling(self, nsamples, verbose = True):
        '''
        Read nsamples from the filehandle
        nsamples is an array or a list
        '''
        sample_stack = self.tif_handle.asarray()[nsamples] # this step is pretty time consuming

        blobs_sample = stack_blobs(sample_stack, self.cdiam, bg_sub = 40)
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
            si = self.stack_cut[0]
            for n_step in range(self.n_groups):
                sf = self.stack_cut[n_step+1]
                substack = self.tif_handle.asarray()[np.arange(si, sf).astype('uint16')] # this is a pretty risky approach, hopefully it can work! @_@
                self.CE_dpt.stack_reload(substack, refill = True)
                sub_time_series = self.CE_dpt.stack_signal_propagate(cblobs) # return
                signal_series.append(sub_time_series)
                si = sf
                if verbose:
                    print("Processed step ", n_step)
            # now, let's concatenate the substacks in the list and compile it into a new dataset 
            ts_signal = np.concatenate(tuple(signal_series), axis = 0)

        else: # one-time load
            raw_stack = self.tif_handle.asarray()
            self.CE_dpt.stack_reload(raw_stack)
            ts_signal = self.CE_dpt.stack_signal_propagate(cblobs)

        self.ts_dataset = position_signal_compile(cblobs, ts_signal)
        np.savez(fname_stem, **self.ts_dataset)
        self.tif_handle.filehandle.close()
        self.tif_handle.close() # close the tif handle 
        self.tif_handle = None
        self.process_log[self.current_file] = 3


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
    data_rootpath ='D:/Data/2018-06-07/\\'
    folder_list = glob.glob(data_rootpath+"/TS_registration\\")
    for data_path in folder_list:
        print(data_path)
        pt = pipeline_tstacks(data_path, fname_flags = 'rg')
        pt.run_pipeline([5,10,15,20])
    #pt = pipeline_tstacks(data_path2, fname_flags = 'ZP')
    #pt.run_pipeline([5,10,15,20])


if __name__ == '__main__':
    main()
