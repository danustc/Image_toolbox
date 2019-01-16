"""
Created by Dan in July-2016
Cell extraction based on the blobs_log in skimage package
Last update: 08/11/2018
The class is supposed to have nothing to do with file name issue. I need to address it out of the class.
"""
import numpy as np
from itbx.preprocessing.red_detect import redund_detect_merge
from skimage import filters, restoration
from skimage.feature import blob_log
from itbx.shared_funcs.numeric_funcs import circ_mask_origin, circ_mask_patch_group, circ_mask_patch


OL_blob = 0.8

# let's have some small handy functions
def blank_refill(raw_frame, cutoff = None, mode = 'min'):
    '''
    If there are zero pixels, refill them.
    if there are too many zero pixels, refill them by sampling
    mode to select:
        min: the mininum N non-zero values
        peak: fill with value around the background
    '''
    if cutoff is None:
        hist, be = np.histogram(raw_frame.ravel(), bins = 256)
        val_ind = hist.argmax() +1
        cutoff = be[val_ind]

    rb_y, rb_x = np.where(raw_frame < cutoff)
    if rb_y.size ==0:
        return raw_frame
    else:
        if mode == 'min':
            raw_valid = np.sort(raw_frame[np.nonzero(raw_frame)])
            nblank = len(rb_y)
            fill_values = raw_valid[:nblank]
            np.random.shuffle(fill_values)
        elif mode == 'peak':
            print("cutoff:", cutoff)
            fill_values = np.random.poisson(lam = cutoff, size = rb_y.size)
        try:
            raw_frame[rb_y, rb_x] = fill_values
        except ValueError as err:
            print(err.args)
            print("The number of zero-pixels exceeded the number of non-zero pixels.")

        return raw_frame

def build_psf(sig, width = 10):
    xx = np.arange(-width, width+1)
    g = np.exp(-xx**2/(2*sig*sig))
    psf = np.outer(g,g)
    psf /= psf.sum()
    return psf


def frame_deblur(raw_frame, sig = 4., Nit = 21, padding = ((10,10), (10,10))):
    '''
    raw_frame: a single frame of image
    sig: the width of gaussian filter
    '''
    print("Initial size:",raw_frame.shape)
    psf = build_psf(sig)
    img = np.pad(raw_frame, padding, mode = 'constant')
    img = blank_refill(img, cutoff=180,  mode = 'peak')

    dc_image = restoration.richardson_lucy(img, psf, iterations = Nit, clip = False)
    ci, cf = padding
    yi, xi = ci
    yf, xf = cf
    recrop = dc_image[yi:-yf, xi:-xf]
    print("Final size:", recrop.shape)
    return blank_refill(recrop, mode = 'peak')

def frame_blobs(filled_frame, bsize = 9, btolerance = 3, bsteps =7, verbose = True, edge_ratio = 3.):
    '''
    extract blobs from a single frame. Added on 06/10/2017
    cblob: a 3-column array, (y, x, sigma), the blob radius is sqrt(2)*sigma
    '''
    # now, let's calculate threshold
    NY, NX = filled_frame.shape
    #th = (np.mean(filled_frame) - np.std(filled_frame))/7. # This is not quite reliable
    th = 32.0
    mx_sig = bsize + btolerance
    mi_sig = bsize - btolerance
    cblobs = blob_log(filled_frame,max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=bsteps, threshold = th, overlap = OL_blob)
    # clean blobs that is at the edge
    by, bx = cblobs[:,0], cblobs[:,1]
    valid_indY = np.logical_and(by > edge_ratio*bsize, by< (NY-edge_ratio*bsize))
    valid_indX = np.logical_and(bx > edge_ratio*bsize, bx< (NX-edge_ratio*bsize))
    valid_ind = np.logical_and(valid_indY, valid_indX)
    cblobs = cblobs[valid_ind] # remove those edge blobs --- added by Dan on 08/11/2018.

    if verbose:
        #print("threshold:", th)
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


def stack_reextract(raw_stack, coords, dr = None):
    '''
    A new version of the function. Instead of recalculating the ROI patches each time, Let's calculate it only once.
    rois: a list of coordinates
    '''
    crs = coords[:,[0,1]]
    if dr is None:
        dr = coords[:,2].min() #Take the mininum of the radius
    #crs = coords[:,:2]
    #dr = coords[:,2].min() #Take the mininum of the radius
    #nz, ny, nx = raw_stack.shape
    YC, XC = circ_mask_patch_group(crs, dr)
    real_sig = raw_stack[:, YC.astype('int'), XC.astype('int')].mean(axis = 2)

    return real_sig


def stack_reextract_old(raw_stack, coords):
    '''
    Assume that the coordinates of the cells have been specified, extract all the cells stackwise
    This is way too slow. Updated on 11/06/2018.
    '''
    crs = coords[:, :2]
    dr = coords[:,2].min() #Take the mininum of the radius

    n_cells = len(coords)
    nz, ny, nx = raw_stack.shape
    real_sig = np.zeros((nz, n_cells))
    for nc in range(n_cells):
        cr = coords[nc,:2]
        #dr = coords[nc,2]-1
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


def stack_blobs(small_stack, diam, sig = 4.5):
    '''
    just extract all the blobs from a small stack and return as a list after background subtraction
    btolerance is always considered 2
    '''
    blobs_stack = []
    for sample_slice in small_stack:
        if sig > 0:
            print("Deblur the stack first.")
            db_slice = frame_deblur(sample_slice, sig )
            cs_blobs = frame_blobs(db_slice, bsize = diam)
        else:
            cs_blobs = frame_blobs(sample_slice.astype('float64'), bsize = diam)
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


    def stack_blobs(self, sig=4., verbose = True):
        """
        process all the frames inside the stack and save the indices of frames containing blobs in self.valid_frames
        Update on 08/16: make the radius of blobs uniform.
        Update on 08/18: merge with the stack_signal_integ
        """
        n_total = 0
        for n_frame in range(self.n_slice):
            '''
            extract blobs from each frame and save into data_list
            '''
            im_raw = self.stack[n_frame]
            if sig > 0:
                print("Deblur the frame first.")
                im0 = frame_deblur(im_raw, sig)
                cblobs = frame_blobs(im0, self.diam)
            else:
                cblobs = frame_blobs(im_raw, self.diam)
            signal_int = frame_reextract(im_raw, cblobs)
            n_blobs = cblobs.shape[0]
            if n_blobs > 0:
                data_slice = np.zeros((n_blobs, 5))
                data_slice[:,2] = n_frame # z-coordinate
                data_slice[:,[0,1,3]] = cblobs # x,y coordinate and diameter
                data_slice[:,4] =signal_int # signal intensity
                n_total += n_blobs

                kwd = 's_'+ str(n_frame).zfill(3)
                self.data_list[kwd] = data_slice

            if verbose:
                print("# blobs in slice", n_frame, ": ", n_blobs)
        self.n_total = n_total

    def zd_bloblist_compression(self, coord_only = True):
        '''
        compress the dictionary into a python nparray
        '''
        n_start = 0
        n_end = 0
        if coord_only:
            d_compression = np.zeros((self.n_total, 3))
            for key, val in self.data_list.items():
                n_end += val.shape[0]
                d_compression[n_start:n_end] = val[:,[2,0,1]]
                n_start = n_end
        return d_compression


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

