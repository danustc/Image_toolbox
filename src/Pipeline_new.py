"""
Created by Dan on 08/15/2016, a test of data processing pipeline
Last update: 06/11/2017, some major changes are made.
Although the low-frequency background is subtracted, cell extraction is still performed on the uncorrected image. This may help eliminating artifacts.
"""
package_path = '/home/sillycat/Programming/Python/Image_toolbox/'

import os
import sys
import glob
import numpy as np
sys.path.append(package_path)
import src.preprocessing.tifffunc as tifffunc

from src.Cell_extract import Cell_extract, frame_reextract

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
    def __init__(self, folder_path, tpflags = 'ZD', offset = 1):
        self.work_folder = folder_path # folder path should be a folder
        self.tif_list = glob.glob(self.work_folder+ '*'+ tpflags +'*.tif')
        self.tif_list.sort(key = os.path.getmtime)
        self.tp_flag = -np.ones(len(self.tif_list)).astype('int16') # all zero, if any time point is converted, then the element is set to True.
        self.dims = None
        name_base = path_leaf(self.tif_list[0])[:-4]
        parse_name = name_base.split('_') # get an array of list that
        self.prefix_z = ''.join(parse_name[:-1]) + '_' # the prefix of the filename
        self.offset = offset
        # OK, time to extract all the flags
        iflag = 0
        for zs_name in self.tif_list:
            zs_tail = path_leaf(zs_name)[:-4]
#             TP_number = int(zs_tail.split('_')[-1]) # get the time point number
            TP_number = number_strip(zs_tail, delim='_', ext = False)
            self.tp_flag[iflag] = TP_number
            iflag += 1

        if(np.any(self.tp_flag<0) == True):
            print("File name mistake. please check! ")
        else:
            print("Initialization completed! All the time-point numbers are extracted.")
            print("z-stack orders:", self.tp_flag)
            self.n_TP = len(self.tp_flag) # number of time points
            self.pro_flag = np.zeros(self.n_TP).astype('bool') # an all-false array to mark t


    def zstack_prepro(self, list_num = 0, deblur = 30, align = True):
        """
        preprocess single zstack.
        list_num: the number in the tif_list
        deblur: non-positive --- no deblur; >0 --- use as the Gaussian filter width
        align: drift align or not?

        """
        zs_name = self.tif_list[list_num]
        print(self.tp_flag[list_num])
        postfix_num = format(self.tp_flag[list_num], '03d') # take out the time point number
        raw_stack = np.copy(tifffunc.read_tiff(zs_name)).astype('float64')


        if(deblur > 0):
            prefix = 'db_'
            print(zs_name)
            z_DB = Deblur(raw_stack, sig = 30) # deblur
            z_DB.stack_high_trunc() # return a new stack with inplane-background subtracted
            z_dbstack = z_DB.get_stack()

            if(self.dims is None):
                self.dims = z_DB.px_num # here we get self.dims
        else:
            prefix = ''
            z_dbstack = raw_stack

        if(align):
            prefix += 'al_'
            ofst = self.offset
            z_DC = Drift_correction(z_dbstack)
            z_DC.drift_correct(offset=ofst, ref_first = False)
            z_newstack = z_DC.get_stack()
            z_drift = z_DC.get_drift()
        else:
            z_newstack = z_dbstack

#         if(prefix != ''):
#             z_writename = self.work_folder + self.prefix_z+prefix + postfix_num+'.tif'
#             tifffunc.write_tiff(z_newstack, z_writename)

        z_CE = Cell_extract(z_newstack)
        z_CE.stack_blobs(msg = True)
        coord_list = z_CE.get_coordinates()


        for zkey, zvalue in coord_list.items():
            """
            # recalculate the blobs average fluorescence from the original
            # instead of shifting the frames, let's shift the center of blobs!
            # zvalue: the y,x coordinates and the radius
            Block updated by Dan on 10/20. Need to be tested.
            """
            z_frame = int(zkey[2:]) # convert the zkey from string into integer
            drift_value = z_drift[z_frame]
            raw_frame = raw_stack[z_frame]
            zvalue[:,0] += drift_value[0]
            zvalue[:,1] += drift_value[1]  # here the coord_list has been changed.

            real_sig = frame_reextract(raw_frame, zvalue)
            z_CE.signal_update(real_sig, zkey)

#             raw_frame = np.roll(raw_frame, -drift_value[0], axis = 0)
#             raw_frame = np.roll(raw_frame, -drift_value[1], axis = 1)


        z_CE.save_data_list(self.work_folder+self.prefix_z+postfix_num) # save as npz
        self.pro_flag[list_num] = True

        return postfix_num # just for information, this returning is useless.
        # done with zstack_prepro

    def zstack_tseries(self, deblur = 30, align = True):
        """
        preprocess all the single zstacks. Should t-alignment be carried out here?
        0. Regardless of the orders in self.tp_flag prepreocess all the zstacks
        1. Reconstruct t-stacks based on the TP number, do the drift correction, realign as t-stacks. (must save the number of drifts.)
        2. Correct the cell positions accordingly, resave as '.npz'. (That part is kinda tricky.)

        """
        for iflag in np.arange(self.n_TP):
            postfix_num = self.zstack_prepro(iflag, deblur, align) # process all the
            print("Processed z point:", postfix_num)


        print("All done.")

#--------------------------------the counterpart for tstacks--------------------------------

class pipeline_tstacks(object):
    '''
    Goal: process all the t-stacks in a folder
    0. Initialize: load the folder and load the files
    1. extract cells and calculate the signals in each t-stack. If t-stack is too large, do this in steps.
    2. Concatenate the processed key-values in a large dataset, then save.
    3. Report progress.
    '''

    def __init__(self, work_folder, fname_flags = '_ZP_'):
        '''
        work_folder: the folder that contains all the .tif files
        '''
        self.work_folder = work_folder
        self.raw_list = glob.glob(work_folder + '*'+fname_flags + '*.tif')
        self.current_file = None # which data set am I working on?

        Nfile = len(self.raw_list)
        if (Nfile ==0):
            print("Error! The folder does not contain any files.")
        else:
            print("There are", Nfile, "image stacks to be processed.")
            self.process_log= np.zeros(Nfile)
            '''
            process_log:    0 --- unprocessed
                            1 --- processed
                            -1 --- error
            '''
            self.CE_dpt = Cell_extract(im_stack = None, diam = 9) # the default diameter of the blob: 9
        return

    def load_file(self, nfile):
        # load the nfileth data file.
        self.current_file = self.raw_list[nfile]



    def sampling(self,)


    def process(nfile, size_th = 500):
        '''
        If size_th > 500 GB, load stepwize
        '''
        self.current_file = self.raw_list[nfile]
        fname = self.current_file
        stack_size, nslices = tifffunc.tiff_describe(fname)
        if(stack_size > size_th):
            # the stack is too large to read. 
            n_groups = int(stack_size/size_th) + 1
            ss_slices = int(nslices/n_groups)+1 # number of slices to import everytime
            stack_cut = np.zeros(n_groups + 1)
            stack_cut[:n_groups] = np.arange(nslices, step = ss_slices)
            stack_cut[-1] = nslices # the cutting-off positions
            si = stack_cut[0]
            for n_step in range(n_groups):
                sf = stack_cut[n_step+1]
                substack = tifffunc.read_tiff(fname, nslice = np.arange(si, sf))
                '''
                An extraction class should be initialized in the __init__ function. However, the stack can remain empty and loaded later.

                '''
                si = sf



        else:
            raw_stack = tifffunc.read_tiff(fname)




