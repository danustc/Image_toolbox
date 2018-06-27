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
#package_path ='/c/Users/Admin/Documents/GitHub/Image_toolbox/src/' # for windows
global_datapath = '/home/sillycat/Programming/Python/data_test/'
portable_datapath = '/media/sillycat/DanData/'
#global_datapath = '/d/Data/Stacks_2b_registered/'
regist_path = '/home/sillycat/Programming/Python/Image_toolbox/cmtkRegistration/'
cluster_path = global_datapath + 'Liquid_delivery/Responsive_clusters/'
pxl_img = [0.295, 0.295, 1.00]
pxl_lab = [0.798, 0.798, 2.00]

def scr_padding():
    '''
    pad all the ZD stacks with zeros and resave
    '''
    ZD_list = glob.glob(portable_datapath+'Jul26*/*ZD*.tif')
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
    #data_path = '/home/sillycat/Programming/Python/cmtkRegistration/'
    ref_path = 'rfp_temp.tif'

    ref_stack = tf.read_tiff(global_datapath+ref_path)
    rm_yaxis = coord_trans.rotmat_yaxis(40.0)
    pxl_img = [0.295, 0.295, 1.00]
    pxl_lab = [0.798, 0.798, 2.00]
    origin_shift = [240, 310, 80]
    origin_shift_MB = [540, 310, 80]
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    sample_range_MB = np.array([1450, 1050, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    ref_range = np.array([138, 621, 1406])
    sample_value = coord_trans.sample_from_refstack(ref_stack, sample_range_MB, pxl_lab, pxl_img, rm_yaxis, origin_shift_MB)
    tf.write_tiff(sample_value, global_datapath + 'RFP_midbrain.tif' )
    print("done!")


def scr_coord_toref():
    '''
    transform the coordinate into those in the reference frame
    has been tested on the RFP template.
    '''
    mark_stat = np.zeros([294, 2])
    MD = maskdb.mask_db()
    resp_list_a = glob.glob(global_datapath+ 'Good_registrations/Apr*_homo.npz')
    resp_list_b = glob.glob(global_datapath+ 'Good_registrations/Apr*_het.npz')
    rm_yaxis = coord_trans.rotmat_yaxis(40.0)
    origin_shift = [240, 310, 80]
    ref_range = np.array([138, 621, 1406])
    sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.
    mask_th = 10
    N_tot = 0
    for coord_file in resp_list_a:
        print(coord_file)
        #coord = np.loadtxt(coord_file)
        data = np.load(coord_file)
        coord = data['coord']
        n_cells = coord.shape[0]
        N_tot += n_cells
        lab_coord = coord_trans.sample_to_refstack_list(coord, sample_range, pxl_img, pxl_lab, rm_yaxis,origin_shift )
        #for rr in lab_coord:
         #   idr = spheri_mask_patch(ref_range, (rr*pxl_lab)[::-1], 3., np.array(pxl_lab[::-1]))
          #  zidx = np.where(idr[:,0]<138)[0]
           # mark_stack[idr[zidx,0], idr[zidx,1], idr[zidx,2]] = 1000


        for n_mask in range(294):
            mask_status = MD.mask_multi_direct_search(n_mask, np.fliplr(lab_coord))
            if np.any(mask_status):
                n_covered = np.sum(mask_status)
                mark_stat[n_mask,0] += n_covered
                print(n_mask, n_covered)
                mask_idx, outline_idx = MD.get_mask(n_mask)
                name_idx = MD.get_name(n_mask)
                print(name_idx)
    mark_stat[:,0] = mark_stat[:,0]/N_tot
    N_tot = 0
    for coord_file in resp_list_b:
        print(coord_file)
        #coord = np.loadtxt(coord_file)
        data = np.load(coord_file)
        coord = data['coord']
        n_cells = coord.shape[0]
        N_tot += n_cells
        lab_coord = coord_trans.sample_to_refstack_list(coord, sample_range, pxl_img, pxl_lab, rm_yaxis,origin_shift )
        #for rr in lab_coord:
         #   idr = spheri_mask_patch(ref_range, (rr*pxl_lab)[::-1], 3., np.array(pxl_lab[::-1]))
          #  zidx = np.where(idr[:,0]<138)[0]
           # mark_stack[idr[zidx,0], idr[zidx,1], idr[zidx,2]] = 1000


        for n_mask in range(294):
            mask_status = MD.mask_multi_direct_search(n_mask, np.fliplr(lab_coord))
            if np.any(mask_status):
                n_covered = np.sum(mask_status)
                mark_stat[n_mask,1] += n_covered
                print(n_mask, n_covered)
                mask_idx, outline_idx = MD.get_mask(n_mask)
                name_idx = MD.get_name(n_mask)
                print(name_idx)

    mark_stat[:,1] = mark_stat[:,1]/N_tot
    np.save(global_datapath + 'Good_registrations/mask_geno', mark_stat)

def scr_mask_compare():
    MD = maskdb.mask_db()
    mark_stat = np.load(global_datapath + 'Good_registrations/mask_geno.npy')
    mark_stat[0] = 0 # Diencephalon:remove this label
    mark_stat[274] = 0 # Telencephalon-general :remove this label
    mark_stat[14] = 0 # Diencephalon-habenula: remove this label
    mark_stat[290] = 0 # Telencephalon-subpallium: remove this label
    mark_stat[282] = 0 # Telencephalon-pallium: remove this label
    mark_stat[285] = 0 # Telencephalon-subpallial gad: remove this label

    ms_sum = mark_stat.sum(axis = 1)
    ind_nz = np.where(ms_sum > 0.01)[0] # thresholding 


    #mark_stat = mark_stat[ind_nz]# the effective mark_stat
    neff_masks = len(ind_nz)
    region_names = []

    for iz in ind_nz:
        mask_name = MD.get_name(iz).split('-')
        region_names.append(mask_name[0][:4]+ mask_name[-1][:15])


    region_a = np.array(region_names)
    fig_a = plt.figure(figsize = (11,6))
    ax= fig_a.add_subplot(111)
    ax.bar(np.arange(neff_masks)-0.20, mark_stat[ind_nz,0],width = 0.40, color = 'coral')
    ax.bar(np.arange(neff_masks)+0.20, mark_stat[ind_nz,1], width = 0.40, color = 'darkviolet')
    fig_a.subplots_adjust(left = 0.12,bottom = 0.23, top = 0.95 )
    ax.set_xticks(np.arange(neff_masks))
    print(region_names)
    ax.set_xticklabels(region_a, rotation = 45, ha = 'right', fontsize = 12)
    #ax.legend(['homo', 'het'])
    ax.legend(['homo (6 fish)', 'het (7 fish)'])
    fig_a.savefig(global_datapath + 'anatomy_geno.png')


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
        tf.write_tiff(sample_value, data_path + simp_key + '.tif')
    hf.close()
    print("done!")

if __name__ =='__main__':
     #scr_mask_compare()
     scr_padding()
