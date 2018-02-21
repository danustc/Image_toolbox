'''
z-brain viewr, python wrapper.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/src/')
import numpy as np
import h5py
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from shared_funcs import tifffunc as tf
from preprocessing import stack_operations as stkop

anatomy_path='/home/sillycat/Programming/Python/cmtkRegistration/AnatomyLabelDatabase.hdf5'
anatomy_folder = '/home/sillycat/Programming/Python/cmtkRegistration/'


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


class Zbrain(object):
    def __init__(self):
        self._load_anatomy_()

    def _load_anatomy_(self):
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



    def close(self):
        self.hf.close()



def main():
    ZB = Zbrain()
    RFP_key = ZB.get_key(12)
    print(RFP_key)
    RFP_template = ZB.access(RFP_key)
    tf.write_tiff(RFP_template, anatomy_folder+'RFP_temp.tif')
    GCaMP_key = ZB.get_key(11)
    print(GCaMP_key)
    GCaMP_template = ZB.access(GCaMP_key)
    tf.write_tiff(GCaMP_template, anatomy_folder+'GCaMP_temp.tif')
    ZB.close()


if __name__ == '__main__':
    main()
