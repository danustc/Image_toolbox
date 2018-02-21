import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import h5py
import numpy as np
import os
import src.shared_funcs.numeric_funcs as numfunc


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


    def get_mask(self, n_mask):
        mstart = self.mask_jc[n_mask]
        mend = self.mask_jc[n_mask+1]

        mask_idx = self.mask_ir[mstart:mend]
        ostart = self.outline_jc[n_mask]
        oend = self.outline_jc[n_mask+1]
        outline_idx = self.outline_ir[ostart:oend]

        return mask_idx, outline_idx

    def mask_multi_coord(self, n_mask, coord, sa_div = [10, 20]):
        '''
        assume that the coordinate has been transformed into the z-brain template frame and ordered as (x,y,z).
        sa_div: solid angle division, default: theta (0,pi) divided into 10 bins while phi (0. 2*pi) divided into 20 bins
        '''
        ncoord = coord.shape[1]
        mask_status = []
        cos_theta_bin = np.linspace(-1, 1, sa_div[0]+1)
        phi_bin = np.linspace(0, np.pi*2, sa_div[1]+1)
        mask_idx, outline_idx = self.get_mask(n_mask)
        mask_z, mask_y, mask_x = np.unravel_index(mask_idx,zb_shape)
        mask_coord = np.c_[mask_x, mask_y, mask_z]
        for blob_coord in coord:
            theta, phi = numfunc.solid_angle(mask_coord - blob_coord)
            H, ct_edges, ph_edges = np.histogram2d(np.cos(theta), phi, bins = (cos_theta_bin, phi_bin))
            print(H.shape)
            if(np.all(H)):
                mask_status.append(True)
            else:
                mask_status.append(False)
        return mask_status


    def shutdown(self):
        self.db.close()
