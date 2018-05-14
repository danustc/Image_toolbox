import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import src.shared_funcs.tifffunc as tf
import src.preprocessing.coord_transform as coord_trans
import src.preprocessing.stack_operations as stack_operations
import src.registration.maskdb_parsing as maskdb
from src.shared_funcs.numeric_funcs import circ_mask_patch, spheri_mask_patch
#---------------Some global variables--------------------
package_path ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/' # for windows
#global_datapath = '/home/sillycat/Programming/Python/data_test/'
global_datapath = '/d/Data/2018-05-07/'
regist_path = '/home/sillycat/Programming/Python/Image_toolbox/cmtkRegistration/'
cluster_path = global_datapath + 'Liquid_delivery/Responsive_clusters/'
pxl_img = [0.295, 0.295, 1.00]
pxl_lab = [0.798, 0.798, 2.00]

def scr_padding():
    '''
    pad all the ZD stacks with zeros and resave
    '''
    ZD_list = glob.glob(global_datapath+'May*/*ZD*.tif')
    print(ZD_list)
    for ZD_file in ZD_list:
        basename = ''.join(ZD_file.split('/')[-2].split('_'))
        ZD_stack = tf.read_tiff(ZD_file)
        zz, zy, zx = np.where(ZD_stack==0)
        ZD_stack[zz,zy,zx] = np.std(ZD_stack)/2.0 + np.random.randn(len(zz))*10
        #TH_stack = stack_operations.stack_global_thresholding(ZD_stack, nsig = th_sig)
        tf.write_tiff(ZD_stack, global_datapath+basename+'.tif')


def scr_fromref():
    # test slice splitting function
    data_path = '/home/sillycat/Programming/Python/cmtkRegistration/'
    ref_path = 'refbrain/rfp_temp.tif'

    ref_stack = tf.read_tiff(data_path+ref_path)
    rm_yaxis = rotmat_yaxis(40.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.798, 0.798, 2.00]
    origin_shift = [240, 310, 80]
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    ref_range = np.array([138, 621, 1406])
    sample_value = coord_trans.sample_from_refstack(ref_stack, sample_range, pxl_lab, pxl_img, rm_yaxis, origin_shift)
    tf.write_tiff(sample_value, data_path + 'RFP_temp.tif' )
    print("done!")


def scr_coord_toref():
    '''
    transform the coordinate into those in the reference frame
    has been tested on the RFP template.
    '''
    mark_stack = np.zeros([138,621,1406])
    mark_stat = np.zeros(294)
    MD = maskdb.mask_db()
    resp_list = glob.glob(global_datapath+ 'Good_registrations/*ref.txt')
    rm_yaxis = coord_trans.rotmat_yaxis(40.0)
    origin_shift = [240, 310, 80]
    ref_range = np.array([138, 621, 1406])
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    for coord_file in resp_list:
        print(coord_file)
        coord = np.loadtxt(coord_file)
        lab_coord = coord_trans.sample_to_refstack_list(coord, sample_range, pxl_img, pxl_lab, rm_yaxis,origin_shift )
        for rr in lab_coord:
            idr = spheri_mask_patch(ref_range, (rr*pxl_lab)[::-1], 3., np.array(pxl_lab[::-1]))
            zidx = np.where(idr[:,0]<138)[0]
            mark_stack[idr[zidx,0], idr[zidx,1], idr[zidx,2]] = 1000


        for n_mask in range(294):
            mask_status = MD.mask_multi_direct_search(n_mask, np.fliplr(lab_coord))
            if np.any(mask_status):
                n_covered = np.sum(mask_status)
                mark_stat[n_mask] += n_covered
                print(n_mask, n_covered)
                mask_idx, outline_idx = MD.get_mask(n_mask)
                name_idx = MD.get_name(n_mask)
                print(name_idx)
    np.save(global_datapath + 'Good_registrations/mask_stat', mark_stat)
    tf.write_tiff(imstack=mark_stack,fname = global_datapath + 'markpxl.tif')
    ind_nz = np.where(mark_stat>0)[0]
    print(ind_nz)
    mark_stat = mark_stat[ind_nz]# the effective mark_stat
    neff_masks = len(mark_stat)
    hneff = int(neff_masks//2)
    print(hneff)
    region_names = []
    for iz in ind_nz:
        mask_name = MD.get_name(iz).split('-')
        region_names.append(mask_name[0][:4]+ mask_name[-1])


    inz_sorted = np.argsort(mark_stat)
    print(inz_sorted)
    region_a = np.array(region_names)[inz_sorted[:hneff]]
    region_b = np.array(region_names)[inz_sorted[hneff:]]
    fig_a = plt.figure(figsize = (9,7))
    ax= fig_a.add_subplot(111)
    ax.bar(np.arange(hneff), mark_stat[inz_sorted[:hneff]])
    fig_a.subplots_adjust(left = 0.2,bottom = 0.35, top = 0.95 )
    ax.set_xticks(np.arange(hneff))
    print(region_names)
    ax.set_xticklabels(region_a, rotation = 50, ha = 'right', fontsize = 12)
    fig_a.savefig(global_datapath + 'anatomy_a.png')

    fig_b = plt.figure(figsize = (9,7))
    ax= fig_b.add_subplot(111)
    ax.bar(np.arange(hneff, neff_masks), mark_stat[inz_sorted[hneff:]])
    fig_b.subplots_adjust(left = 0.2, bottom = 0.35, top = 0.95 )
    ax.set_xticks(np.arange(hneff, neff_masks))
    print(region_names)
    ax.set_xticklabels(region_b, rotation = 50, ha = 'right', fontsize = 12)
    fig_b.savefig(global_datapath + 'anatomy_b.png')

def scr_tempcrop():
    data_path = '/home/sillycat/Programming/Python/cmtkRegistration/'
    ref_path = 'AnatomyLabelDatabase.hdf5'

    hf = h5py.File(data_path + ref_path, 'r')
    kl = list(hf.keys())
    print(kl)
    rm_yaxis = rotmat_yaxis(40.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.798, 0.798, 2.00]
    origin_shift = [240, 310, 80]
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    ref_range = np.array([138, 621, 1406])
    for temp_key in kl[3:]:
        ref_stack = np.array(hf[temp_key])
        simp_key = '_'.join(temp_key.split('_')[:-1])
        sample_value = sample_from_refstack(ref_stack, sample_range, pxl_lab, pxl_img, rm_yaxis, origin_shift)
        tf.write_tiff(sample_value, data_path + simp_key + '.tif')
    hf.close()
    print("done!")

if __name__ =='__main__':
     scr_padding()
