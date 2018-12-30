import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import tifffile as tf
import itbx.registration.coord_transform as coord_trans
import itbx.registration.maskdb_parsing as maskdb
from itbx.shared_funcs.numeric_funcs import circ_mask_patch, spheri_mask_patch
#---------------Some global variables--------------------

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/'
global_datapath_win = 'E:\\/2018-10-11/'
#sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
portable_datapath = '/media/sillycat/DanData/'
regist_path = '/home/sillycat/Programming/Python/Image_toolbox/cmtkRegistration/'
pxl_img = [0.295, 0.295, 1.00]
pxl_lab = [0.798, 0.798, 2.00]

def scr_padding():
    '''
    pad all the ZD stacks with zeros and resave
    '''
    ZD_list = glob.glob(global_datapath_win+'Oct*/*ZD*.tif') # This is for ubuntu
    #ZD_list = glob.glob(global_datapath+'Jun*/*ZD*.tif') # This is for windows 
    print(ZD_list)
    for ZD_file in ZD_list:
        basename = os.path.basename(ZD_file)
        dirname = os.path.dirname(ZD_file)
        base_dir = ''.join(os.path.basename(dirname).split('_'))
        print(base_dir)
        ZD_stack = tf.imread(ZD_file)
        zz, zy, zx = np.where(ZD_stack==0)
        ZD_stack[zz,zy,zx] = np.abs(np.random.randn(len(zz))*15)
        #TH_stack = stack_operations.stack_global_thresholding(ZD_stack, nsig = th_sig)
        tf.imsave(global_datapath_win+base_dir+'.tif', ZD_stack.astype('uint16'))


def scr_fromref():
    # test slice splitting function
    #data_path = '/home/sillycat/Programming/Python/cmtkRegistration/'
    ref_path = 'rfp_temp.tif'

    ref_stack = tf.imread(global_datapath+ref_path)
    rm_yaxis = coord_trans.rotmat_yaxis(36.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.798, 0.798, 2.00]
    origin_shift = [240, 310, 80]
    origin_shift_MB = [540, 310, 115]
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    sample_range_MB = np.array([1050, 1450, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    ref_range = np.array([138, 621, 1406])
    sample_value = coord_trans.sample_from_refstack(ref_stack, sample_range_MB, pxl_lab, pxl_img, rm_yaxis, origin_shift_MB)
    tf.imwrite(global_datapath + 'RFP_midbrain.tif', sample_value )
    print("done!")


def scr_tempcrop():
    data_path = '/home/sillycat/Programming/Python/cmtkRegistration/'
    ref_path = 'AnatomyLabelDatabase.hdf5'

    hf = h5py.File(data_path + ref_path, 'r')
    kl = list(hf.keys())
    print(kl)
    rm_yaxis = coord_trans.rotmat_yaxis(40.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.798, 0.798, 2.00]
    origin_shift = [240, 310, 80]
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    ref_range = np.array([138, 621, 1406])
    for temp_key in kl[3:]:
        ref_stack = np.array(hf[temp_key])
        simp_key = '_'.join(temp_key.split('_')[:-1])
        sample_value = sample_from_refstack(ref_stack, sample_range, pxl_lab, pxl_img, rm_yaxis, origin_shift)
        tf.imsave(data_path + simp_key + '.tif', sample_value.astype('uint16'))

    hf.close()
    print("done!")




if __name__ =='__main__':
     #scr_coord_toref()
     scr_padding()
