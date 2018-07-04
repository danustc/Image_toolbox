import os
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import glob
import numpy as np
import subprocess
import time
import pandas as pd
import calendar
import src.preprocessing.coord_transform as coord_trans
#-------------------Global variables-----------------
exu = "cmtk streamxform"
data_path="/home/sillycat/Programming/Python/data_test/"
meta_path = data_path + 'metadata_sheet.csv'
month_num = {v: k for k,v in enumerate(calendar.month_abbr)}

pxl_img = [0.295, 0.295, 1.00]
pxl_lab = [0.798, 0.798, 2.00]
origin_shift = [240, 310, 80]
ref_range = np.array([138, 621, 1406])
sample_range = np.array([976, 724, 120]) # the sample range is ordered reversely w.r.t the stack shape, i.e., x--y--z.

def anatomical_labeling(coord_file):
    '''
    transform the coordinate into those in the reference frame, then annotate each cell
    coord_file: the file (reformatted) to be labeledannatomically.
    '''
    dest_path = coord_file.split('.')[0]+'_lb'
    MD = maskdb.mask_db()
    rm_yaxis = coord_trans.rotmat_yaxis(40.0) # rotational matrix
    print(coord_file)
        #coord = np.loadtxt(coord_file)
    data = np.load(coord_file)
    coord = data['coord']
    n_cells = coord.shape[0]
    mask_labels = np.zeros(n_cells, 294)
    annotation_labels = np.zeros(n_cells)
    lab_coord = coord_trans.sample_to_refstack_list(coord, sample_range, pxl_img, pxl_lab, rm_yaxis,origin_shift )


    for n_mask in range(294):
        mask_labels[:,n_mask] = MD.mask_multi_direct_search(n_mask, np.fliplr(lab_coord))
        mask_idx, outline_idx = MD.get_mask(n_mask)
        name_idx = MD.get_name(n_mask)
        print(name_idx)

    #next, label each cell 
    mask_dup = np.sum(mask_labels, axis = 1) # for each cell, sum up the masks to count how many brain regions are annotated
    for ii in range(n_cells):
        if mask_dup[ii]>1:
            '''
            this cell is labeled by more than one
            '''
            mdup = mask_dup[ii]
            m_labeled = np.where(mask_labels[ii] > 0)[0]
            MM = 0
            for jj in range(mdup):
                MM+=294**(mdup-1-jj)*m_labeled[jj]

            annotation_labels[ii] = MM

        elif mask_dup[ii] ==1:
            MM = np.argmax(mask_labels[ii])
            annotation_labels[ii] = MM

        else:
            annotation_labels[ii] = -1


      labeled_dataset = dict()
      labeled_coord = np.zeros([n_cells, 4])
      labeled_coord[:,:3] = coord
      labeled_coord[:,3] = annotation_labels
      labeled_dataset['coord'] = labeled_coord
      labeled_dataset['signal'] = data['signal']

      np.savez(dest_path, **labeled_dataset)






def dimension_check(key_date, meta_dim, nyear = '17'):
    # parse the key_data first 
    if key_date[0].isalpha():
        month = key_date[:3]#
        n_month = format(month_num[month], '02')
        date = key_date[3:5]
        comp_key = n_month+date+nyear+'_'+key_date[-2:]
        print(comp_key)
    else:
        comp_key = key_date
    try:
        key_dim = meta_dim.ix[comp_key]
    except IndexError:
        print("index error.")

    return key_dim['NX']



def registered_coord_convert(rg_path, coord_source, coord_dest, inv = True):
    '''
    convert a list of coordinates from the unregistered frame into the registered frame.
    '''
    if inv:
        command = exu + ' -- --inverse %s < %s > %s'
    else:
        command = exu + ' %s < %s > %s'
    #subprocess.call([self.executable, files_file, str(self.multiplier),
    subprocess.Popen(command%(rg_path, coord_source, coord_dest), shell = True)


def coord_convert_preprocess(fpath, reg_list, origin_x = 975, order = 'r'):
    '''
    Convert the unregistered coordinates into registered coordinates
    origin_x: the original x length. I need to have a database for this.
    order: if the original coordinate order is 'x-y-z', then order = 'f'; otherwise order = 'r'.
    '''
    raw_data = np.load(fpath)
    basename = '_'.join(os.path.basename(fpath).split('.')[0].split('_')[:-1])
    parent_path= os.path.dirname(fpath)
    coord = raw_data['coord']
    if order == 'r':
        coord[:,2] = origin_x*0.295-coord[:,2] # This needs to be validated.
        coord = coord[:,::-1]
    else:
        coord[:,0] = origin_x*0.295-coord[:,0]
    try:
        signal = raw_data['data'].T
    except KeyError:
        try:
            signal = raw_data['signal'].T
        except KeyError:
            return -1
    source_name = parent_path + '/' + basename + '_coord.txt'
    dest_name = parent_path + '/' + basename + '_rawref.txt'
    finedest_name = parent_path + '/' + basename + '_ref'  # the destination filename of .npz
    NC = coord.shape[0]
    if os.path.isdir(reg_list):
        np.savetxt(source_name, coord, fmt = '%12.5f')# save the coordinate in the x,y,z order.  
        registered_coord_convert(reg_list, source_name, dest_name, inv = True)
        time.sleep(5) # wait until the file is saved
        f = open(dest_name, 'r')
        raw_coord = f.readlines()
        f.close()
        fine_coord = []
        fine_data = []
        nf = 0
        # here I add some correction for FAILED coordinates.
        for nc in range(NC):
            s = raw_coord[nc]
            d = signal[nc]
            if 'F' in s:
                nf +=1
            else:
                sn = [float(ii) for ii in s[:-2].split(' ')]
                fine_coord.append(sn)
                fine_data.append(d)
        ref_data = dict()

        ref_data['coord'] = np.array(fine_coord) # back to z-y-x
        print(len(fine_coord))
        ref_data['signal'] = np.array(fine_data).T
        np.savez(finedest_name, **ref_data)
        time.sleep(1)
    else:
        print('The registration parameter does not exist.')

    return 0



def outer_shell(thickness = 10):
    '''
    take out the outer shell of the dataset.
    '''
    data_list = glob.glob(data_path+'Good_registrations/Jul2017_rest/*_ref.npz')

    for data_name in data_list:
        dset = np.load(data_name)
        coord = dset['coord']
        signal = dset['signal']
        xx = coord[:,0]
        yy = coord[:,1]
        zz = coord[:,2]
        xmin = np.min(xx)
        xmax = np.max(xx)
        ymin = np.min(yy)
        ymax = np.max(yy)
        zmin = np.min(zz)
        zmax = np.max(zz)

        core_x = np.logical_and(xx-xmin>thickness, xmax-xx>thickness)
        core_y = np.logical_and(yy-ymin>thickness, ymax-yy>thickness)
        core_z = np.logical_and(zz-zmin>thickness, zmax-zz>thickness)
        core_xy = np.logical_and(core_x, core_y)
        core_xyz = np.logical_and(core_xy, core_z)
        coord_shell = coord[~core_xyz]
        signal_shell = signal[:, ~core_xyz]

        shell_dataset = dict()
        shell_dataset['coord'] = coord_shell
        shell_dataset['signal'] = signal_shell
        fname = data_name[:-4] + '_shell'
        np.savez(fname, **shell_dataset)





def main():
    meta_df = pd.read_csv(meta_path, sep = ',')
    meta_dim = meta_df[['Fish','NY', 'NX']]
    meta_dim.set_index('Fish', inplace = True)
    response_list = glob.glob(data_path+'Good_registrations/Jul2017_rest/*_dff.npz')
    print(response_list)
    for response_file in response_list:
        basename ='_'.join( os.path.basename(response_file).split('.')[0].split('_')[:-1])

        print("Fish:", basename)
        xdim =dimension_check(basename, meta_dim)
        print(xdim)
        temp_list = basename.split('_')
        reglist  = ''.join([temp_list[0], temp_list[-1]])

        reg_list = data_path + 'Good_registrations/Jul2017_rest/' + reglist +  '.list'
        sta = coord_convert_preprocess(response_file,reg_list,int(xdim),order = 'r')


if __name__ == '__main__':
    main()
