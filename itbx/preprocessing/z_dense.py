"""
Last Modification: 04/26/2017 by Dan.
"""
import os
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import numpy as np
from src.algos.linked_list import Simple_list
from src.shared_funcs.numeric_funcs import lateral_distance


def z_dense_construct(zd_file):
    '''
    Construct a z_dense dictionary from the raw z_dense file, add one column to stock redundancy information.
    Column 0: y coordinates
    Column 1: x coordinates
    Column 2: z slices
    Column 3: originally radius, changed to redundancy flags, initialized as 0
    Column 4: intensity (kept for gaussian real z identification)
    '''
    zd = np.load(zd_file)
    z_dense = {}
    for keys, values in zd.items():
        new_entry = values
        new_entry[:,3] = 0
        z_dense[keys] = new_entry

    return z_dense


# ---------------------------Big classes ------------------------------------

class z_dense_ref(object):
    """
    load a densely labeled stack, remove redundancy
    """
    def __init__(self, z_dense, dims, z_step=1.0, verbose = False):
        z_keys = z_dense.keys()
        nz  = len(z_keys)
        self.ny, self.nx = dims
        self.z_dense = z_dense # z_dense: a densely labeled stack containing y-x coordinates, 
        self.nz = nz
        self.z_step=z_step
        self.verbose = verbose
        self.redundancy_pool = {} # create an empty redundancy pool


    def _red_detect_(self, nslice = 0, thresh = 2.0):
        """
        detect redundancy centered around the slice n.
        thresh: threshold for redundancy detection
        """
        zk_1 = 's_' + format(nslice, '03d')
        zk_2 = 's_' + format(nslice+1, '03d')

        zf_1 = self.z_dense[zk_1]
        zf_2 = self.z_dense[zk_2]

        # extract the y and x coordinates
        y1 = zf_1[:,0]
        x1 = zf_1[:,1]

        y2 = zf_2[:,0]
        x2 = zf_2[:,1]


        # create a meshgrid
        [YC, YR] = np.meshgrid(y2, y1)
        [XC, XR] = np.meshgrid(x2, x1)


        dist_block = np.sqrt((YC-YR)**2 + (XC-XR)**2)
        red_pair = np.where(dist_block <= thresh) # find out where the distance between cell i in plane k and cell j in plane k+1 is below the threshold.

        ind1 = red_pair[0] # the indices in the first frame
        ind2 = red_pair[1] # the indices in the second frame


        # select those with markers > 0 and markers < 0
        marker_1 = zf_1[ind1, 3]


        new_idx = (marker_1 == 0) # select those with zero-markers, which are never counted before. These are new cells. marker_1 needs to be updated.
        pool_new = ind1[new_idx] # select the indices in the first frame where new redundancies are detected 
        pool_new_cov = ind2[new_idx] # select the indices in the second frame where new redundancies are detected.


        pool_exist = ind1[~new_idx] # among the detected redundancies, find those already marked.
        pool_exist_cov = ind2[~new_idx] # correspondingly, find those already marked in the adjacent slice

        n_new = len(pool_new)
        n_exist = len(pool_exist)
        if self.verbose:
            print(n_new, "new redundancies, ", n_exist, "existing redundancies")

        for n_count in np.arange(n_new):
            # build the new keys
            # also, we need to assign each new key an identity number which is unique.
            n_ind1 = pool_new[n_count] # find the indices in the first slice that contains new redundancies
            n_ind2 = pool_new_cov[n_count] # find the indices in the following slice 
            pr_number =  nslice * 1000 + n_ind1
            pr_key = 'sl_' + str(pr_number) # build a key 
            new_sl = Simple_list(nslice) # create a simple list with z_marker = nslice, nslice is the index of the first z-slice 
            new_sl.add([nslice, zf_1[n_ind1, 4]])
            new_sl.add([nslice+1, zf_2[n_ind2, 4]])
            zf_1[n_ind1, 3] = pr_number # assign the new pr_number to zf_1
            zf_2[n_ind2, 3] = pr_number # assigne the same new pr_number to zf_2

            self.redundancy_pool[pr_key] = new_sl # stored into the redundancy pool


        for n_count in np.arange(n_exist):
            # search for the existing keys
            n_ind1 = pool_exist[n_count]
            n_ind2 = pool_exist_cov[n_count]
            pr_number = int(zf_1[n_ind1, 3])# catch up the pr_number
            pr_key = 'sl_' + str(pr_number) # this pr_key should already exist in the pool. 

            self.redundancy_pool[pr_key].add([nslice+1, zf_2[n_ind2, 4]])
            zf_2[n_ind2, 3] = pr_number # update the pr_number in the adjacent slice

#         return ind1, ind2  # return the indices which indicates the marked positions of redundancy
        # end of paired redundancy detection



    def stack_red_detect(self):
        """
        detect the whole stack's redundancy and correct them.
        steps:
        0. Create an empty dictionary to store the link list
        1. compare the first two frames (z0, z1) and detect the redundancy, return the indices
        2. create a couple of link lists that have two nodes, set all the "updated" status as False.
        3. compare the next two frames (z1, z2) and detect the redundancy again. If the cells of z1 are unmarked in the last comparison,
        initialize a new linklist; otherwise find the identity key (should be a string) and append the frame number to the old list.
        """
        self.redundancy_pool.clear()

        for nslice in np.arange(self.nz-1):
            self._red_detect_(nslice, thresh = 1.0)

        # OK, let's check the the size of the pool and remove them one by one.
        dist_3d = np.zeros((0, 4)) # create an empty array to save z, y, x, f


        for sl_key, sl_value in self.redundancy_pool.items():
            z_start = sl_value.z_marker # where does the z_marker starts
            z_list = np.array(sl_value.list) # convert it into a 2d array
            z_key = 's_' + format(z_start, '03d')
            zframe_0 = self.z_dense[z_key]
            z_identifier = int(sl_key[3:]) - z_start*1000 # which cell?

            pz = self.z_step*np.inner(z_list[:,0], z_list[:,1])/z_list[:,1].sum() # weighted average estimation
            py, px = zframe_0[z_identifier, 0:2] # The x-y coordinates
            pf = zframe_0[z_identifier, 4] # the fluorescence


            new_entry = np.array([[pz, py, px, pf]])
            dist_3d = np.concatenate((dist_3d, new_entry), axis = 0)

        ord_z = np.argsort(dist_3d[:,0], axis = 0)
        # sort in the order of Z.


        self.dist_3d = dist_3d[ord_z, :]

        return dist_3d
    # ----Done with redundancy removal



    def frame_zalign(self, zf_coord, z_init = 0.0, search_range = 10.0, thresh = 1.50 ):
        """
        Purpose: align a frame's cell with 3-D reconstructed, redundancy removed distributions.
        zf_coord: the coordination of cells on the frame that is to be aligned.
        search_range: in which z-range should we search for the cells?
        thresh: threshold of lateral distance in the unit of pixels
        return:
        """
        dist_3d = self.dist_3d # catch the dist_3d
        z_coord = dist_3d[:,0] # taking out the z coordinates of all the

        z_upper = z_init+search_range*0.5
        z_lower = z_init-search_range*0.5
        z_index = np.where(np.logical_and(z_coord > z_lower, z_coord < z_upper))[0]# taking out the indices
        print("Cells to be identified:", zf_coord.shape)
        search_block = dist_3d[z_index, :] # only search for the search block
        z_block = search_block[:,0]
        search_lateral = search_block[:,[1,2]] # exclude the z-column and the fluorescence column
        dR = lateral_distance(search_lateral, zf_coord)

        red_pair = np.where(dR <= thresh)

        ind1 = red_pair[0] # the indices of search_lateral
        ind2 = red_pair[1] # the indices of zf_coord

        # This is a messy but magic block to detect ambiguities :D
        uind2, unique_indices, unique_counts = np.unique(ind2, return_index = True, return_counts = True) # if more than one cells are identified with each cell on zf_coord
        print(len(uind2),"out of",  len(ind2), "are unique.")
        n_detected = len(ind2)
        zf_3d = np.zeros((n_detected, 3)) #
        zf_3d[:,1:] = zf_coord[ind2,:]
        zf_3d[:,0] =  z_block[ind1]

        amb_count = np.where(unique_counts>1)[0] #
        if(amb_count.size): # there are ambiguities
            z_ambind = uind2[amb_count] # select which indices are overcounted
            # iterate through z_ambind
            for za in z_ambind:
                ref_ind = np.where(ind2==za)
                ref_z = z_block[ind1[ref_ind]]
                z_closest = np.min(np.abs(ref_z-z_init))
                zf_3d[ref_ind, 0] = z_closest
        # then, remove the duplication

            zfu_ind = np.unique(zf_3d[:,0], return_index = True)[1] # only return the indices
            zf_3d = zf_3d[zfu_ind, :] # this is sorted in z-direction

        return zf_3d



# ----------------------------Test functions---------------------
def main():
    fpath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'


if __name__ == '__main__':
    main()
