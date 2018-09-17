import os
import sys
import glob
import numpy as np
sys.path.append(package_path_ubn)

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
        self.CE_dpt.stack_blobs(sig = 4, verbose = True)

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


