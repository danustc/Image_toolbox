"""
Created by Dan in July-2016
Cell extraction based on the blobs_log in skimage package
Last update: 06/09/2017
Now it really feels lousy. :( Try to use as few for loops as you can!
The class is supposed to have nothing to do with file name issue. I need to address it out of the class.
"""
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import random
import matplotlib.pyplot as plt
from src.preprocessing.tifffunc import read_tiff
from src.preprocessing.Red_detect import redund_detect_merge
from skimage import filters
from skimage.feature import blob_log
from skimage.filters import threshold_local
from src.shared_funcs.numeric_funcs import circ_mask_patch


OL_blob = 0.5
magni_lateral = 0.295 # 0.295 micron per pixel

# let's have some small handy functions
def frame_deblur(raw_frame, sig = 40, ofst=0):
    '''
    raw_frame: a single frame of image
    sig: the width of gaussian filter
    '''
    raw_valid = np.nonzero(raw_frame).sort()
    rb_y, rb_x = np.where(raw_frame==0)
    nblank = len(rb_y)
    rb_fill = random.shuffle(raw_valid[:nblank])
    raw_frame[rb_y, rb_x] = rb_fill
    ifilt = filters.gaussian(raw_frame, sigma = sig)
    iratio = raw_frame/ifilt
    nmin = np.argmin(iratio)
    gmin_ind = np.unravel_index(nmin, raw_frame.shape)

    sca =raw_frame[gmin_ind]/ifilt[gmin_ind]
    db_frame = raw_frame - ifilt*sca*0.999
    return db_frame


def frame_reextract(raw_frame, coords):
    """
    Let's make it simple: instead of extracting from the whole stack, just extract from one frame.
    So life's gonna be much easier!
    """
    f_dims = raw_frame.shape
    n_cells = len(coords) # number of cells

    real_sig = np.zeros(n_cells)

    for nc in np.arange(n_cells):
        cr = coords[nc,:2] # the real center on non-drift corrected frame
        dr = coords[nc,2]
        indm = circ_mask_patch(f_dims, cr, dr)
        real_sig[nc] = np.mean(raw_frame[indm]) # is it OK for replacing dr with

    return real_sig

# adjust the positions of cells 
def cell_list_afm(clist, afm, afb):
    '''
    Affine transformation of a list of cell positions.
    clist: the list of extracted cells (y-x positions only)
    afm: the matrix of the affine transformation
    afb: the shift vector of the affine transformation.
    '''
    xy_coord = np.fliplr(clist).T # transform the matrix
    n_cells = len(clist) # number of the cells
    b_tile = p.tile(afb,(n_cells, 1)).T
    tc_list = np.matmul(afm, xy_coord) +b_tile
    return tc_list


#-----------------------------------------------------------------------------------
class Cell_extract(object):
    # this class extracts
    def __init__(self, im_stack, diam = 6):
        self.stack = im_stack
        self.data_list = {}
        self.n_slice = im_stack.shape[0]
        self.bl_flag = np.zeros(self.n_slice).astype('int') # create an all-zero array for
        self.frame_size = np.array(im_stack.shape[1:])
        self.blobset = [diam+1, diam-1, diam]


    def image_blobs(self, n_frame):
        """
        Extract number of blobs from the frame n_frame.
        Updated: self.data_list
        """

        im0 = self.stack[n_frame]
        im_ept = np.where(im0 == 0)
        VI = np.floor(np.std(im0)/10) # variability of the pixels 
        im0[im_ept] = np.random.randint(VI)+10
        mx_sig = self.blobset[0]
        mi_sig = self.blobset[1]
        nsig = self.blobset[2]
        # comment on 09/05: we need a smarter way to do the threshold setting.
        block_size = 55
        th = (np.mean(im0)-np.std(im0))/5.0
        print("threshold:", th)
        cblobs = blob_log(im0,max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL_blob)

        n_blobs = cblobs.shape[0]
        if(n_blobs == 0):
            raise ValueError("This slice contains no blobs or has not been processed yet. ")
            self.bl_flag[n_frame] = -1
            return
        else:
            frame_size = self.frame_size
            self.bl_flag[n_frame] = n_blobs
            data_slice = np.empty([n_blobs, 5]) # initialize an empty array
            # also, remove blobs that are on the margin

            for ii in np.arange(n_blobs):
                blob = cblobs[ii]
                # going through all blobs in list
                cr = blob[0:2]
                dr = blob[-1]
#                 mask = circ_mask([self.ny, self.nx], cr, dr)
                mask = circ_mask_patch(frame_size, cr, dr)
                signal_int = im0[mask].mean() # replace sum() with mean()
                data_slice[ii] = np.array([blob[0], blob[1], n_frame, dr, signal_int])

        # also, update self.data_list here instead of in stack_blobs, so you can use it right after single-frame processing!
        kwd = 's_'+ str(n_frame).zfill(3)
        self.data_list[kwd] = data_slice
        return data_slice
        # done with image_blobs


    def stack_blobs(self, msg = False):
        """
        process all the frames inside the stack and save the indices of frames containing blobs in self.valid_frames
        Update on 08/16: make the radius of blobs uniform.
        Update on 08/18: merge with the stack_signal_integ
        """

        for n_frame in np.arange(self.n_slice):
            self.image_blobs(n_frame)
            if msg:
                n_blobs = self.bl_flag[n_frame].astype('int64')
                print("number of blobs in %d th frame: %d" %(n_frame, n_blobs))

        self.valid_frames = np.where(self.bl_flag>0)[0]
        # end of the function stack_blobs

    def extract_sampling(self, nsamples, mode = 'm', bg_sub = 40):
        '''
        nsamples: indice of slices that are selected for cell extraction
        mode:   m --- mean of the selected slices, then extract cells from the single slice
                a --- extract cells from all the slices and do the redundance removal
                bg_sub:if true, subtract background.
        '''
        single_slice = False
        ext_stack=self.stack[nsamples]
        if(mode == 'm'):
            mean_slice = np.mean(ext_stack,axis = 0)
            single_slice = True
        if(bg_sub > 0):
            # subtract the background
            if single_slice:
                db_slice = frame_deblur(mean_slice, bg_sub)
            else:
                for sample_slice in ext_stack:
                    db_slice = frame_deblur(sample_slice, bg_sub)
                    ext_stack[]





    def stack_signal_propagate(self, n_frame = 0, verbose = False):
        """
        Assume that all the slices are aligned and morphologically the same. We only extract cells
        from one slice (usually the first one), and integrate the values at the same sites at the rest
        slices.This method should not be used in z-stacks.
        Procedures:
        0 --- calculate the blobs in the first slice
        1 --- replace the radius with the mininum radius
        2 --- assign cblobs to the clists
        3 --- image_signal_integ
        Update on 08/19. Output: a dict. ['xy']: coordinates; ['data']: fluorescence signal.
        """
        if(np.isscalar(n_frame)):
            if self.bl_flag[n_frame]>0: # if the cell extraction is already done
                kwd = 's_'+ str(n_frame).zfill(3)
                data_slice = self.data_list[kwd]
            else:
                data_slice = self.image_blobs(n_frame) # extract blobs from the selected frame first
            n_blobs = self.bl_flag[n_frame]
        else:
            # Added on 04/13/2017           
            # if we have multiple slices
            data_ref = self.image_blobs(n_frame[0])
            print("Initial extraction:",data_ref.shape)
            for nf in n_frame[1:]:
                # extract cells from each frame, detect redundancy, then merge
                data_fol = self.image_blobs(nf)
                ind_ref, ind_fol = redund_detect_merge(data_ref,data_fol, thresh = 5)
               # merge the two slice; append the redundant ones behind 
                nr_ref = len(ind_ref)
                nr_fol = len(ind_fol)
                mask_ref = np.array([ir in ind_ref for ir in np.arange(len(data_ref))])
                mask_fol = np.array([io in ind_fol for io in np.arange(len(data_fol))])
                dr_unique = data_ref[~mask_ref] # the unique part in data_ref
                df_unique = data_fol[~mask_fol] # the unique part in data_fol
                if(np.isscalar(ind_ref)):
                    data_redund = np.array([data_ref[ind_ref, :]])
                else:
                    data_redund = data_ref[ind_ref,:]
                data_merge = np.concatenate((dr_unique, df_unique, data_redund), axis = 0 )
                data_ref = data_merge
                if verbose:
                    print("t_slice:", nf)
                    print("direct extraction:", data_fol.shape)
                    print("level of redundancy:", data_redund.shape)
                    print("unique slice nblobs:", dr_unique.shape, df_unique.shape)
                    print("overall nblobs:", data_merge.shape)
            n_blobs = len(data_ref)# merging extracted cells in several frames
            data_slice = data_ref
            # ----- end else, n_frame is an array instead of a slice number

        blob_time_stack = dict()
        coords = data_slice[:,:2] # takenout the y and x coordinates as maps
        blob_time_stack['xy'] = coords
        dr_min = np.min(data_slice[:,3])-0.5 # get an uniform dr.
        train_signal = np.zeros((self.n_slice, n_blobs))

        for z_frame in np.arange(self.n_slice):
            self.bl_flag[z_frame] = n_blobs
            z_signal = self.image_signal_propagate(z_frame, coords, dr_min)
            train_signal[z_frame, :] = z_signal

        blob_time_stack['data'] = train_signal

        return blob_time_stack
    # done with stack_signal_propagate


    def image_signal_propagate(self,z_frame, maps, dr):
        """
        Added on 08/18 to replace image_signal_integ.
        The idea is similar to that in the image_blobs
        maps: the (y,x) coordinates
        return: only fluorescence instead of coordinate and fluorescence.
        """
        im0 = self.stack[z_frame]
        frame_size = self.frame_size
        nblobs = maps.shape[0]
        f_slice = np.zeros(nblobs)
        ii = 0
        for coord in maps:
            mask = circ_mask_patch(frame_size, coord, dr)
            f_slice[ii] = im0[mask].mean()
            ii += 1

        return f_slice


    def save_data_list(self, dph):
        """
        Presumption: self.data_list has been fully updated
        dph: data path + file name
        """
        np.savez(dph, **self.data_list) # with keys saved


    def get_coordinates(self):
        # Only return the y-x coordinates of the blobs; ignore everything else
        data_list = self.data_list
        coord_list = {}

        for zkey, zvalue in data_list.items():
            coord_list[zkey] = zvalue[:,[0,1]] # take out the y, x coordinates and the radius

        return coord_list


    def signal_update(self, new_sig, nkey):
        """
        Updates the signal of the nth frame while keep the coordinates.
        """
        self.data_list[nkey][:,-1] = new_sig



    def stack_reload(self, new_stack):
        """
        Updates the image stack saved in the class, reset everything
        """
        self.stack = new_stack
        self.n_slice, ny, nx = new_stack.shape
        self.frame_size = np.array([ny,nx])
        self.bl_flag = np.zeros(self.n_slice)
        self.data_list.clear()
        print("reload completed.")
        # ----- reload the im_stack


def main():
    '''
    The main function for testing the cell extraction code.
    '''
    tf_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    TS_slice9 = 'TS_folder/rg_A1_FB_TS_ZP_9.tif'
    TS_slice14 = 'TS_folder/rg_A1_FB_TS_ZP_14.tif'
    ZD_stack = 'A1_FB_ZD.tif'
    zstack = read_tiff(tf_path+ZD_stack).astype('float64')
    CEz = Cell_extract(zstack)
    CEz.stack_blobs(msg=True)
    CEz.save_data_list(tf_path+'A1_FB_ZD')
# 
    tstack9 = read_tiff(tf_path+TS_slice9).astype('float64')
    tstack14 = read_tiff(tf_path+TS_slice14).astype('float64')
    CEt = Cell_extract(tstack9)
    bt_stack = CEt.stack_signal_propagate(n_frame = np.arange(5), verbose = True)
    np.savez(tf_path+'TS_9', **bt_stack)
    CEt.stack_reload(tstack14) # reload the stack 14
    bt_stack = CEt.stack_signal_propagate(n_frame = np.arange(5), verbose = True)
    np.savez(tf_path+'TS_14', **bt_stack)

if __name__ == '__main__':
    main()
