"""
Created on 05/06/2017, for indirect registration between different T-stacks through a ZD-stack.
Test affine transformation
"""


import os
from src.preprocessing.z_dense import z_dense_ref, z_dense_construct
import numpy as np
import glob
import src.preprocessing.tifffunc as tf
import src.preprocessing.Affine as Affine
from src.visualization.brain_navigation import slice_display,stack_display
from src.preprocessing.Red_detect import redund_detect_merge

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/High_Quality_FB_Redundancy_removed/'



def Coord_read_transform(fn_trans, fn_data):
    '''
    coordinate transform based on the MultiStackReg outputs.
    then, resave the data.
    '''
    raw_data = np.load(fn_data)
    coord, fluo = raw_data['xy'], raw_data['data']
    afm, afv = Affine.aff_read(fn_trans, average = True)
    rfm, rfv= Affine.reverse_trans(afm, afv)
    afc= np.fliplr(Affine.pixel_transform(np.fliplr(coord), rfm, rfv))

    return afc, fluo


# ----------------------------Test functions-------------------------


def dumb1(ariz_list = [], rota_list = []):
    '''
    1.load all the TS data for the slices
    '''
    indi_list = glob.glob(global_datapath+'*')
    for indi_folder in indi_list:
        print(indi_folder)
        reglist = glob.glob(indi_folder+"/*.txt")
        datalist = glob.glob(indi_folder+"/*ZP*.npz")
        nreg = np.array([int(os.path.basename(regfile).split('.')[0].split('_')[-1] ) for regfile in reglist ])
        ndata = np.array([int(os.path.basename(datafile).split('.')[0].split('_')[-1]) for datafile in datalist])
        arg_reg = np.argsort(nreg)
        arg_data = np.argsort(ndata)
        reglist=[reglist[i] for i in arg_reg]
        datalist = [datalist[i] for i in arg_data]
        if(np.any(nreg[arg_reg]-ndata[arg_data])):
            print("There are files missing or index error!")
            return
        else:
            n_all = 0
            nmark = len(nreg)
            reg_ref = reglist[0]
            data_ref = datalist[0]
            afc_ref, fluo_ref = Coord_read_transform(reg_ref, data_ref)
            afc_merge = {}
            afc_backup = {}
            fluo_merge = {}
            fluo_backup = {}
            for imark in np.arange(1, nmark):
                # the redundant cells are contained in the ref slices, while deleted from the cor slices and saved as a back-up.
                data_cor = datalist[imark]
                reg_cor = reglist[imark]
                afc_cor, fluo_cor = Coord_read_transform(reg_cor, data_cor)
                ind_ref, ind_cor = redund_detect_merge(afc_ref, afc_cor, thresh = 3.0)
                label_m = 'merge_'+format(imark-1, '03d')
                bk_m = 'back_'+format(imark-1, '03d')
                ncell = afc_ref.shape[0]
                n_all+=ncell
                cr_3 = np.zeros((ncell,3))
                cr_3[:, 0] = (imark-1)*4.
                cr_3[ind_ref,0] = (2*imark-1)*2 # for the redundant cells, set the z-position in the middle
                cr_3[:,1:] = afc_ref
                # make a subarray of unique cells in the extended coordinate list
                afc_reduced = np.delete(afc_cor, ind_cor, axis = 0)
                fluo_reduced = np.delete(fluo_cor, ind_cor, axis = 1) # note the difference of axis
                afc_backup[bk_m] = afc_cor[ind_cor]
                fluo_backup[bk_m] = fluo_cor[ind_cor]
                afc_merge[label_m] = cr_3
                fluo_merge[label_m] = fluo_ref
                afc_ref = afc_reduced
                fluo_ref = fluo_reduced
            # end for
            ncell = afc_ref.shape[0]
            n_all+=ncell
            cr_3 = np.zeros((ncell,3))
            cr_3[:, 0] = (imark-1)*4.
            cr_3[:,1:] = afc_ref
            label_m = 'merge_'+format(imark-1, '03d')
            bk_m = 'back_'+format(imark-1, '03d')
            afc_merge[label_m] = cr_3
            fluo_merge[label_m] = fluo_ref
            np.savez(indi_folder[:-2]+'merge_zyx', **afc_merge)
            np.savez(indi_folder[:-2]+'merge_fluo',**fluo_merge)
            np.savez(indi_folder[:-2]+'backup_zyx', **afc_backup)
            np.savez(indi_folder[:-2]+'backup_fluo',**fluo_backup)



if __name__ == '__main__':
    dumb1()# 
