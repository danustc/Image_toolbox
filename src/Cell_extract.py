"""
Created by Dan in July-2016
Cell extraction based on the blobs_log in skimage package
Last update: 10/19/16
Now it really feels lousy. :( Try to use as few for loops as you can!
The class is supposed to have nothing to do with file name issue. I need to address it out of the class.
"""
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.tifffunc import read_tiff
from preprocessing.Red_detect import redund_detect_merge
from skimage.feature import blob_log
from shared_funcs.numeric_funcs import circ_mask_patch


OL_blob = 0.5
magni_lateral = 0.295 # 0.295 micron per pixel

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
        """

        im0 = self.stack[n_frame]
        mx_sig = self.blobset[0]
        mi_sig = self.blobset[1]
        nsig = self.blobset[2]
        # comment on 09/05: we need a smarter way to do the threshold setting.
        th = (np.mean(im0)-np.min(im0))/18.0
        # th = 0.50
        plt.imshow(im0)
        plt.savefig('test')

        cblobs = blob_log(im0,
            max_sigma = mx_sig, min_sigma = mi_sig, num_sigma=nsig, threshold = th, overlap = OL_blob)


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



            # This sounds really lousy.

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
                ind_ref, ind_fol = redund_detect_merge(data_ref,data_fol, thresh = 4)
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
                    print("unique slice nblobs:", dr_unique.shape, df_unique.shape)
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
        """
        Only return the coordinates and diameters of the blobs; lose the intensity
        """
        data_list = self.data_list
        coord_list = {}

        for zkey, zvalue in data_list.items():
            coord_list[zkey] = zvalue[:,[0,1,3]] # take out the y, x coordinates and the radius

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
    TS_slice9 = 'A1_FB_TS_ZP_9.tif'
    tstack = np.copy(read_tiff(tf_path+TS_slice9).astype('float64'))
    CE = Cell_extract(tstack)
    blob_time_stack = CE.stack_signal_propagate([0,1,2],verbose = True)
    np.savez(tf_path+'test_z9_s2', **blob_time_stack)
    print("CE class initialized.")


if __name__ == '__main__':
    main()
