'''
The overall pipeline for image processing
Created by Dan on 04/28/2018, an updated version of the image processing pipeline. Can be run in terminal or called by the UI.
'''

package_path ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/'
#package_path = '/home/sillycat/Programming/Python/Image_toolbox/src/' # This is Ubuntu version
import os
import sys
import glob
import numpy as np
sys.path.append(package_path)
import src.shared_funcs.tifffunc as tf
import src.preprocessing.segmentation as segmentation
import src.preprocessing.drift_correction as drc
import src.preprocessing.crossalign_pipeline as crossalign


class pipeline(object):
    '''
    This pipeline imports an image stack and does the processing accordingly.
    will search for:
        all the .tif files in the working folder
        all the .npz files in the working folder
    '''

    def __init__(self, fpath = None, cdiam = 9, flag = 'ZP'):
        '''
        fpath: the directory where the raw images are stored.
        '''
        self.work_folder = fpath
        self.CE = segmentation.Cell_extract(im_stack = None, diam = cdiam)
        self.loaded = False
        self.flag = flag
        self.cdiam = cdiam
        self.parse_workfolder()

    def parse_workfolder(self, ext = '*.tif'):
        '''
        parsing the work folder.
        '''
        self.raw_list = glob.glob(self.work_folder + '*'+self.flag +ext)
        self.Nfile = len(self.raw_list)
        print(self.raw_list)

    def load_file(self, nfile, size_th = 600, verbose = True):
        '''
        Load the nfileth data file.
        size_th: threshold of the data file. Unit: MB.
        '''
        fname = self.raw_list[nfile]
        stack_size, stack_shape, tif_handle = tf.tiff_describe(fname, handle_open = True)
        nz = stack_shape[0]
        if(stack_size > size_th):
            self.gigantic = True
            n_groups = int(stack_size//size_th)+1
            if (nz%n_groups ==0):
                l_sec = int(nz // n_groups)
            else:
                l_sec = int(nz // n_groups) + 1
            if verbose:
                print("number of group:", n_groups)
                print("section length:", l_sec)
        else:
            self.gigantic = False

        self.stack_shape = stack_shape
        self.current_file = nfile
        self.stack = tif_handle.asarray()
        self.tif_handle = tif_handle
        self.stack_cut = np.arange(n_groups)*l_sec
        self.stack_cut[-1] = nz
        self.n_groups = n_groups

    def sampling(self, nsamples, verbose = True):
        '''
        Read nsamples from the filehandle
        nsamples is an array or a list
        '''
        sample_stack = self.tif_handle.asarray()[nsamples] # this step is pretty time consuming

        blobs_sample = segmentation.stack_blobs(sample_stack, self.cdiam, bg_sub = 40)
        self.cblobs = segmentation.stack_redundreduct(blobs_sample, th = 5) # redundancy removed substack, saves the y,x coordinates of the extracted blobs
        if verbose:
            print("Done with sampling! Number of blobs:", self.cblobs.shape[0])


    def segment_sample_propagate(self, verbose = True):
        '''
        suppose the sampling has been done, and current file has been processed.
        If size_th > 500 GB, load stepwize
        this should be run after self.sampling
        '''
        cblobs = self.cblobs
        fname_stem = os.path.splitext(self.raw_list[self.current_file])[0]
        # stepload or one-time load?
        if(self.gigantic):
            signal_series = [] # create an empty list, which should be merged later
            si = self.stack_cut[0]
            for n_step in range(self.n_groups-1):
                sf = self.stack_cut[n_step+1]
                substack = self.tif_handle.asarray()[np.arange(si, sf).astype('uint16')] # this is a pretty risky approach, hopefully it can work! @_@
                sub_time_series = segmentation.stack_signal_propagate(substack, cblobs) # return
                signal_series.append(sub_time_series)
                si = sf
                if verbose:
                    print("Processed step ", n_step)
            # now, let's concatenate the substacks in the list and compile it into a new dataset 
            ts_signal = np.concatenate(tuple(signal_series), axis = 0)

        else: # one-time load
            raw_stack = self.tif_handle.asarray()
            ts_signal = segmentation.stack_signal_propagate(raw_stack, cblobs)

        self.ts_dataset = segmentation.position_signal_compile(cblobs, ts_signal)
        np.savez(fname_stem, **self.ts_dataset)
        self.tif_handle.filehandle.close()
        self.tif_handle.close() # close the tif handle 
        self.tif_handle = None

    def batch_segmentation(self, n_samples, verbose = True):
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
            self.segment_sample_propagate()


    def shutDown(self):
        if self.tif_handle is not None:
            self.tif_handle.close()



def main():
    # data_rootpath ='D:\Data/2017-06-27/A1_GCDA\\' # this is for Windows
    data_rootpath ='/home/sillycat/Programming/Python/data_test/' # this is for Windows
    folder_list = glob.glob(data_rootpath+'*.tif')
    PL = pipeline(fpath = data_rootpath, cdiam = 9, flag = 'ZP')
    PL.load_file(0)
    PL.run_pipeline(n_samples = [0,5,10])



if __name__ =='__main__':
    main()
