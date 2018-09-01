'''
z-brain viewr, python wrapper.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import h5py
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from src.shared_funcs import tifffunc as tf
import maskdb_parsing as maskdb

db_path = '/home/sillycat/Programming/Python/cmtkRegistration/MaskDatabase.mat'
anatomy_path='/home/sillycat/Programming/Python/cmtkRegistration/AnatomyLabelDatabase.hdf5'
data_path="/home/sillycat/Programming/Python/data_test/"

zb_origin_shift = [240, 310, 80]
zb_sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.

def load_mat(mat_path):
    '''
    load matlab data file (*.mat).
    '''
    if os.path.exists(mat_path):
        try:
            im_stack_raw = sio.loadmat(mat_path)['data']
            im_stack = np.transpose(im_stack_raw,(2,0,1))
            return im_stack.astype('float64') # only keep the data of the dictionary
        except IOError:
            print("Could not read the file:", mat_path)
            sys.exit()

# --------------------- Visualization class -----------

class Zbrain(object):
    def __init__(self):
        #self._load_anatomy_()
        self.MD = maskdb.mask_db()
        self.keys = None
        self.hf = None

    def load_anatomy(self):
        self.hf = h5py.File(anatomy_path, 'r')
        self.keys = list(self.hf.keys())

    def access(self, key):
        if key in self.keys:
            data = np.array(self.hf[key])
            return data
        else:
            print("The key is not in the anatomy database.")

    def get_key(self, n = 12):
        try:
            return self.keys[n]
        except IndexError:
            print("Out of bound key index! ")

    def anatomical_compare(self,label_counts, label_names, nc_cut = 10):
        '''
        barplot of masks summary.
        label_counts: NM x 2 array, number of cells under each label.
        '''
        fig_as = plt.figure(figsize = (12,6))
        ax_dist = fig_as.add_subplot(111)
        fig_as.subplots_adjust(bottom = 0.2, left = 0.35)
        ind_sel = label_counts > nc_cut
        ax_dist.bar()
        return fig_as

    def close(self):
        self.MD.shutdown()
        if self.hf is not None:
            self.hf.close()



def main():
    '''
    Summarize the # of neurons in each brain region
    Columns: Mask label (0-294)
    '''
    ZB = Zbrain()
    mask_summ = np.load(data_path + 'FB_resting_15min/Jun07_2018/*ref.npz')
    mask_keys = mask_summ.keys()
    print(mask_keys)
    for key, mask_info in mask_summ.items():
        print(key)
        for mask_label in mask_info:
            name = ZB.MD.get_name(mask_label[0])
            print(mask_label[0], name)


if __name__ == '__main__':
    main()
