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
import matplotlib.pyplot as plt
from src.shared_funcs.tifffunc import read_tiff
from src.preprocessing.red_detect import redund_detect_merge
from skimage import filters
from skimage.feature import blob_log
from src.shared_funcs.numeric_funcs import circ_mask_patch
from src.visualization.brain_navigation import slice_display


OL_blob = 0.2

# let's have some small handy functions
def blank_refill(raw_frame):
    '''
    If there are zero pixels, refill them.
    if there are too many zero pixels, refill them by sampling
    '''
    rb_y, rb_x = np.where(raw_frame==0)
    if rb_y.size ==0:
        return raw_frame
    else:
        raw_valid = np.sort(raw_frame[np.nonzero(raw_frame)])
        nblank = len(rb_y)
        fill_values = raw_valid[:nblank]
        np.random.shuffle(fill_values)
        try:
            raw_frame[rb_y, rb_x] = fill_values
        except ValueError as err:
            print(err.args)
            print("The number of zero-pixels exceeded the number of non-zero pixels.")

        return raw_frame


def frame_deblur(raw_frame, sig = 40 ):
    '''
    raw_frame: a single frame of image
    sig: the width of gaussian filter
    '''
    ifilt = filters.gaussian(raw_frame, sigma = sig)
    iratio = raw_frame/ifilt
    nmin = np.argmin(iratio)
    gmin_ind = np.unravel_index(nmin, raw_frame.shape)
    sca =raw_frame[gmin_ind]/(ifilt[gmin_ind])
    db_frame = raw_frame - ifilt*sca*0.999
    return db_frame

def frame_blobs(filled_frame, bsize = 9, btolerance = 2, bsteps =7, verbose = False):
    '''
    extract blobs from a single frame. Added on 06/10/2017
    cblob: a 3-column array, (y, x, sigma), the blob radius is sqrt(2)*sigma
    '''
    # now, let's calculate threshold
    th = (np.mean(filled_frame) - np.std(filled_frame))/6.
    mx_sig = bsize + btolerance
    mi_sig = bsize - btolerance
    cblobs = blob_log(filled_frame,max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=bsteps, threshold = th, overlap = OL_blob)
    if verbose:
        print("# of blobs:", cblobs.shape[0])
    return cblobs


def frame_reextract(raw_frame, coords):
    """
    Let's make it simple: instead of extracting from the whole stack, just extract from one frame.
    So life's gonna be much easier!
    coords: column 0 --- y coordinate
            column 1 --- x coordinate
            column 2 --- blob size
    """
    f_dims = raw_frame.shape
    n_cells = len(coords) # number of cells

    real_sig = np.zeros(n_cells)

    for nc in range(n_cells):
        cr = coords[nc,:2] # the real center on non-drift corrected frame
        dr = coords[nc,2]-1 # shrink the blob size to reduce the influence brought up by the sample shifts
        indm = circ_mask_patch(f_dims, cr, dr)
        real_sig[nc] = np.mean(raw_frame[indm]) # is it OK for replacing dr with

    return real_sig


def stack_reextract(raw_stack, coords):
    '''
    Assume that the coordinates of the cells have been specified, extract all the cells stackwise
    '''
    n_cells = len(coords)
    nz, ny, nx = raw_stack.shape
    real_sig = np.zeros((nz, n_cells))
    for nc in range(n_cells):
        cr = coords[nc,:2]
        dr = coords[nc,2]-1
        indm = circ_mask_patch((ny,nx), cr, dr)
        real_sig[:,nc] = np.mean(raw_stack[:, indm[0],indm[1]], axis = 1)

    return real_sig



def stack_redundreduct(blob_stack, th= 4):
    '''
    blob_stack: a list of blobs
    thresh: threshold of redundance detection
    '''
    len_stack = len(blob_stack)
    data_ref = blob_stack[0]
    for data_fol in blob_stack[1:]:
        ind_ref, ind_fol = redund_detect_merge(data_ref,data_fol, thresh = th)
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
        n_blobs = len(data_ref)# merging extracted cells in several frames
        data_slice = data_ref

    return data_merge


def stack_blobs(small_stack, diam, bg_sub = 40):
    '''
    just extract all the blobs from a small stack and return as a list after background subtraction
    btolerance is always considered 2
    '''
    blobs_stack = []
    for sample_slice in small_stack:
        db_slice = frame_deblur(sample_slice, bg_sub)
        cs_blobs = frame_blobs(db_slice, bsize = diam)
        blobs_stack.append(cs_blobs)

    return blobs_stack


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


def position_signal_compile(coords, signals):
    '''
    a small function compiles the coordinates and the signals (raw F), which forms a dictionary
    '''
    blob_time_stack = dict()
    blob_time_stack['xy'] = coords
    blob_time_stack['data'] = signals
    return blob_time_stack






#-----------------------------------------------------------------------------------
class Cell_extract(object):
    # this class extracts
    def __init__(self, im_stack = None, diam = 9):
        '''
        im_stack: None or a fully loaded stack
        '''
        if im_stack is None:
            print("No stack is loaded.")
            self.is_empty = True
        else:
            self.stack = im_stack
            self.n_slice = im_stack.shape[0]
            self.frame_size = np.array(im_stack.shape[1:])
            self.is_empty = False

        self.data_list = dict()
        self.diam = diam
        self.redund = True


    def image_blobs(self, n_frame, bg_sub):
        '''
        Last update: 06/14/2017.
        Extract number of blobs from the frame n_frame.
        Updated: self.data_list; a data_slice is returned.
        each data_slice is a 5-column array: y, x, z,dr, signal intensity.
        This is only called by the function stack_blobs, which extracts blobs from each slice individually.
        '''
        im0 = frame_deblur(self.stack[n_frame], bg_sub)
        cblobs = frame_blobs(im0, self.diam)
        signal_int = frame_reextract(im0, cblobs)

        n_blobs = cblobs.shape[0]
        frame_size = self.frame_size
        if n_blobs > 0:
            data_slice = np.zeros((n_blobs, 5))
            data_slice[:,2] = n_frame
            data_slice[:,[0,1,3]] = cblobs
            data_slice[:,4] =signal_int

            kwd = 's_'+ str(n_frame).zfill(3)
            self.data_list[kwd] = data_slice
        return n_blobs

    def stack_blobs(self, bg_sub=40, verbose = True):
        """
        process all the frames inside the stack and save the indices of frames containing blobs in self.valid_frames
        Update on 08/16: make the radius of blobs uniform.
        Update on 08/18: merge with the stack_signal_integ
        """
        for n_frame in range(self.n_slice):
            '''
            extract blobs from each frame and save into data_list
            '''
            n_blobs = self.image_blobs(n_frame, bg_sub)
            if verbose:
                print("# blobs in slice", n_frame, ": ", n_blobs)


    def extract_sampling(self, nsamples, mode = 'm', bg_sub = 40, red_reduct = 5):
        '''
        nsamples: indice of slices that are selected for cell extraction
        mode:   m --- mean of the selected slices, then extract cells from the single slice
                a --- extract cells from all the slices and do the redundance removal
                bg_sub:if true, subtract background.
                the core part is rewrapped into an independent function.
        '''
        single_slice = False
        ext_stack= self.stack[nsamples] # copy a few slices and do background subtraction so that more blobs can be extracted.
        n_ext = len(nsamples)
        if(mode == 'm'):
            mean_slice = np.mean(ext_stack,axis = 0)
            single_slice = True
        if(bg_sub > 0):
            # subtract the background
            if single_slice:
                db_slice = frame_deblur(mean_slice, bg_sub)
                cblobs = frame_blobs(db_slice,self.diam)
            else:
                blobs_sample = stack_blobs(ext_stack, self.diam, bg_sub)

                # redundance reduction
                if (red_reduct > 0):
                # do the redundancy removal on the list
                    cblobs = stack_redundreduct(blobs_sample, red_reduct)
                else:
                    return blobs_sample
        return cblobs



    def stack_signal_propagate(self, blob_lists):
        '''
        OMG... This so badly written.
        blob_lists: it contains 3 columns: y, x, dr.
        '''
        stack = self.stack
        n_blobs = len(blob_lists)# merging extracted cells in several frames
            # ----- end else, n_frame is an array instead of a slice number
        train_signal = stack_reextract(stack, blob_lists) # updated on 07/07/2017

        '''
        for z_frame in range(self.n_slice):
            z_signal = frame_reextract(stack[z_frame], blob_lists)
            train_signal[z_frame, :] = z_signal
        '''
        return train_signal
    # done with stack_signal_propagate


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



    def stack_reload(self, new_stack, refill = True):
        """
        Updates the image stack saved in the class, reset everything
        """
        n_slice, ny, nx = new_stack.shape
        if refill: #refill the blank pixels
            ref_stack = np.zeros_like(new_stack).astype('float64')
            for nz in range(n_slice):
                ref_stack[nz] = blank_refill(new_stack[nz].astype('float64'))
            self.stack = ref_stack
        else:
            self.stack = new_stack.astype('float64')
        self.n_slice = n_slice
        self.frame_size = np.array([ny,nx])
        self.redund = True
        self.data_list.clear()
        # ----- reload the im_stack


def main():
    '''
    The main function for testing the cell extraction code.
    '''
    #tf_path_win = 'D:\Data/2017-06-06/A2_TS_Compare\\'
    tf_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    TS_stack = 'TS_folder/rg_A1_TS_Compare_ZP_2.tif'
    tstack = read_tiff(tf_path+TS_stack, np.arange(400)).astype('float64')
    CEt = Cell_extract(tstack)
    ext_blobs = CEt.extract_sampling(nsamples = [4, 5, 6, 7, 8 ], mode = 'a', red_reduct = 6 )
    print(ext_blobs.shape)
    bt_stack = CEt.stack_signal_propagate(ext_blobs)
    np.savez(tf_path+'Compare_test', **bt_stack)
    figd = slice_display(ext_blobs, title = "extraction", ref_image = tstack[100])
    figd.savefig(tf_path + 'extracted_blobs')

if __name__ == '__main__':
    main()
