import h5py
import numpy as np
import os
import itbx.shared_funcs.numeric_funcs as numfunc


db_path = '/home/sillycat/Programming/Python/cmtkRegistration/MaskDatabase.mat'

zb_shape = [138,621,1406]

class mask_db(object):
    def __init__(self):
        self.db = h5py.File(db_path, 'r')
        self.name_list = self.db['MaskDatabaseNames']
        mask_list = self.db['MaskDatabase']
        outline_list = self.db['MaskDatabaseOutlines']
        self.mask_ir = mask_list['ir']
        self.mask_jc = mask_list['jc']
        self.outline_ir = outline_list['ir']
        self.outline_jc = outline_list['jc']

    def get_name(self, n_mask):
        rawname_idx = self.name_list[n_mask][0]
        name_idx = ''.join(chr(i) for i in self.db[rawname_idx][:])
        return name_idx

    def get_mask(self, n_mask):
        mstart = self.mask_jc[n_mask]
        mend = self.mask_jc[n_mask+1]

        mask_idx = self.mask_ir[mstart:mend]
        ostart = self.outline_jc[n_mask]
        oend = self.outline_jc[n_mask+1]
        outline_idx = self.outline_ir[ostart:oend]

        return mask_idx, outline_idx

    def mask_multi_direct_search(self,n_mask, raw_coord):
        '''
        direct search instead of histogram over the 4pi solid angle.
        raw_coord: the coordinates to be labeled
        n_mask: # of mask
        return: a boolean array
        '''
        ncoord = raw_coord.shape[0]
        mask_status = []
        floor_pxls = np.floor(raw_coord).astype('int')
        ceil_pxls = np.ceil(raw_coord).astype('int')
        mask_idx, outline_idx = self.get_mask(n_mask)
        set_mask = set(mask_idx)
        masked = []
        c_000 = np.ravel_multi_index(floor_pxls.T, zb_shape)
        c_111 = np.ravel_multi_index(ceil_pxls.T, zb_shape)
        c_001 = np.ravel_multi_index([floor_pxls[:,0], floor_pxls[:,1], ceil_pxls[:,2]], zb_shape)
        c_011 = np.ravel_multi_index([floor_pxls[:,0], ceil_pxls[:,1], ceil_pxls[:,2]], zb_shape)
        c_101 = np.ravel_multi_index([ceil_pxls[:,0], floor_pxls[:,1], ceil_pxls[:,2]], zb_shape)
        c_100 = np.ravel_multi_index([ceil_pxls[:,0], floor_pxls[:,1], floor_pxls[:,2]], zb_shape)
        c_010 = np.ravel_multi_index([floor_pxls[:,0], ceil_pxls[:,1], floor_pxls[:,2]], zb_shape)
        c_110 = np.ravel_multi_index([ceil_pxls[:,0], ceil_pxls[:,1], floor_pxls[:,2]], zb_shape)
        nearest_vortices = np.c_[c_000, c_100, c_010, c_001, c_110, c_101, c_011, c_111]
        for v in nearest_vortices: # iterate through all the cells
            masked.append(set(v).issubset(set_mask))


        return masked, np.array(masked).sum()

    def mask_name_lite(self, fpath):
        '''
        make a light version of the mask labels
        '''
        with open(fpath, 'w') as f:
            for iname in range(294):
                name_str = self.get_name(iname)
                f.write("%s\n" % name_str)
        # the file will automatically close.

    def shutdown(self):
        self.db.close()
