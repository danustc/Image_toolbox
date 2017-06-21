'''
cross align between a ZD stack and a group of TS stacks.
last update: 06/20/2017
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import os
import numpy as np
import glob
import src.preprocessing.affine as Affine
from src.visualization.brain_navigation import slice_display,stack_display
from src.preprocessing.red_detect import redund_detect_merge
global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'


def Coord_read_transform(fn_trans, fn_data):
    '''
    coordinate transform based on the MultiStackReg outputs.
    then, resave the data.
    '''
    raw_data = np.load(fn_data)
    coord, fluo = raw_data['xy'][:,:2], raw_data['data']
    afm, afv = Affine.aff_read(fn_trans, average = True)
    rfm, rfv= Affine.reverse_trans(afm, afv)
    afc= np.fliplr(Affine.pixel_transform(np.fliplr(coord), rfm, rfv))
    return afc, fluo


def cross_align_simple(work_folder, rg_flag = 'TS2ZD', data_flag = 'ZP', zstep = 4.0, verbose = True):
    '''
    cross align without redundancy removal
    '''
    regilist = glob.glob(work_folder + '*' + rg_flag + '*.txt')
    datalist = glob.glob(work_folder+ '*' + data_flag + '*.npz')
    nregi = np.array([int(os.path.basename(regfile).split('.')[0].split('_')[-1] ) for regfile in regilist ])
    ndata = np.array([int(os.path.basename(datafile).split('.')[0].split('_')[-1]) for datafile in datalist])
    arg_regi = np.argsort(nregi)
    arg_data = np.argsort(ndata)
    try:
        diff_arg = nregi[arg_regi]-ndata[arg_data]
        if(np.any(diff_arg)):
            print("There are files missing or index error!")
            print(arg_regi)
            print(arg_data)
            return
    except ValueError:
        print("Oops! The number of registration files and data files do not match.")
        return
    cr_total = []
    fl_total = []
    data_compiled = dict()

    nmark = len(nregi)
    n_all = 0
    for imark in np.arange(nmark):
        regi_fn = reglist[0]
        data_fn = reglist[0]
        afc, fluo = Coord_read_transform(regi_fn, data_fn)
        ncells = afc.shape[0]
        cr_3 = np.zeros((ncells, 3))
        cr_3[:,0] = imark*zstep
        cr_3[:,1:] = afc
        n_all +=ncells
        cr_total.append(cr_3)
        fl_total.append(fluo)

    data_compiled['coords'] = np.concatenate(cr_total, axis = 0)
    data_compiled['signal'] = np.concatenate(fl_total, axis = 1)

    return data_compiled


def cross_align_folder(work_folder, rg_flag = 'TS2ZD', data_flag = 'ZP', zstep = 4.0, verbose =True):
    '''
    cross align all the T-stacks to the Z-stack based on the affine transformatiomatrices
    '''
    regilist = glob.glob(work_folder + '*' + rg_flag + '*.txt')
    datalist = glob.glob(work_folder+ '*' + data_flag + '*.npz')
    nregi = np.array([int(os.path.basename(regfile).split('.')[0].split('_')[-1] ) for regfile in regilist ])
    ndata = np.array([int(os.path.basename(datafile).split('.')[0].split('_')[-1]) for datafile in datalist])
    arg_regi = np.argsort(nregi)
    arg_data = np.argsort(ndata)
    try:
        diff_arg = nregi[arg_regi]-ndata[arg_data]
        if(np.any(diff_arg)):
            print("There are files missing or index error!")
            print(arg_regi)
            print(arg_data)
            return
    except ValueError:
        print("Oops! The number of registration files and data files do not match.")
        return

    nmark = len(nregi)
    n_all = 0
    regi_ref = regilist[0]
    data_ref = datalist[0]
    afc_ref, fluo_ref = Coord_read_transform(regi_ref, data_ref)
    afc_merge = {}
    afc_backup = {}
    fluo_merge = {}
    fluo_backup = {}
    for imark in np.arange(1, nmark):
        '''
        Affine transformation of each TS slice, then redundancy removal slice by slice.
        '''
        data_cor = datalist[imark]
        regi_cor = regilist[imark]
        afc_cor, fluo_cor = Coord_read_transform(regi_cor, data_cor)
        ind_ref, ind_cor = redund_detect_merge(afc_ref, afc_cor, thresh = 3.0)
        label_m = 'merge_'+ format(imark-1, '03d')
        bk_m = 'back_' + format(imark-1, '03d')
        ncell = afc_ref.shape[0]
        n_all+=ncell
        cr_3 = np.zeros((ncell,3))
        cr_3[:, 0] = (imark-1)*zstep
        cr_3[ind_ref,0] = (2*imark-1)*2
        cr_3[:,1:] = afc_ref
        afc_reduced = np.delete(afc_cor, ind_cor, axis = 0)
        fluo_reduced = np.delete(fluo_cor, ind_cor, axis = 1) # note the difference of axis
        afc_backup[bk_m] = afc_cor[ind_cor]
        fluo_backup[bk_m] = fluo_cor[ind_cor]
        afc_merge[label_m] = cr_3
        fluo_merge[label_m] = fluo_ref
        afc_ref = afc_reduced
        fluo_ref = fluo_reduced
        if verbose:
            print("Processed slice:", imark)
            print("# of redundancy detected:", len(afc_cor))

    ncell = afc_ref.shape[0]
    n_all+=ncell
    cr_3 = np.zeros((ncell,3))
    cr_3[:, 0] = (imark-1)*4.
    cr_3[:,1:] = afc_ref
    label_m = 'merge_'+format(imark-1, '03d')
    bk_m = 'back_'+format(imark-1, '03d')
    afc_merge[label_m] = cr_3
    fluo_merge[label_m] = fluo_ref

    return afc_merge, fluo_merge
    # done with cross_align_folder 


def data_integrate(afc_merge, fluo_merge):
    '''
    put the two aligned coordinates and fluorescence data together to reconstructa 3D representation
    The z-coordinate is not ordered.
    '''
    if (set(afc_merge)==set(fluo_merge)):
        # check if the two dictionaries have the same keys. If yes, keep going
        di_keys = afc_merge.keys()
        coor_list = []
        data_list = []
        for km in di_keys:
            coor_list.append(afc_merge[km])
            data_list.append(fluo_merge[km])

        coor_3d = np.concatenate(coor_list, axis = 0)
        data_3d = np.concatenate(data_list, axis = 1)

        compiled_data = {'coord': coor_3d, 'data': data_3d}
        return compiled_data


# ---------------------------Below is the testing function ---------------------
def main():
    relative_path = 'Jun13_B2_control/'
    full_path = global_datapath+relative_path
    afc_merge, fluo_merge = cross_align_folder(full_path)
    compiled_data = data_integrate(afc_merge, fluo_merge)
    np.savez(full_path+'merged', **compiled_data)

if __name__ == '__main__':
    main()

