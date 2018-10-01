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
import maskdb_parsing as maskdb
import matplotlib.pyplot as plt

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

def edge_cropping(coord_file, edge = 5.0):
    rawd = np.load(coord_file)
    coord = rawd['coord']
    signal = rawd['signal']
    mx, my, mz = np.max(coord, axis = 0)
    xr = np.logical_and(coord[:,0]> edge, coord[:,0]< (mx-edge))
    yr = np.logical_and(coord[:,1]> edge, coord[:,1]< (my-edge))
    zr = np.logical_and(coord[:,2]> edge, coord[:,2]< (mz-edge))
    core_xy = np.logical_and(xr, yr)
    core_xyz = np.logical_and(core_xy, zr)
    edge_coord = coord[~core_xyz]
    edge_signal = signal[:, ~core_xyz]

    edge_data = {'coord': edge_coord, 'signal': edge_signal}
    np.savez(coord_file[:-4]+'_ed', **edge_data)


def anatomical_labeling(coord_file, arti_clear = True):
    '''
    transform the coordinate into those in the reference frame, then annotate each cell
    coord_file: the file (reformatted) to be labeledannatomically.
    arti_clear: remove cells that have anatomical label -1
    '''
    print(coord_file)
    if coord_file[-4] == '.':
        dest_path = coord_file.split('.')[0]+'_lb'
        data = np.load(coord_file)
    else:
        dest_path = coord_file+'_lb'
        data = np.load(coord_file+'.npz')

    MD = maskdb.mask_db()
    rm_yaxis = coord_trans.rotmat_yaxis(40.0) # rotational matrix
    print(coord_file)
        #coord = np.loadtxt(coord_file)
    coord = data['coord'] # the freshly registered coordinates
    signal = data['signal']

    lab_coord = coord_trans.sample_to_refstack_list(coord, sample_range, pxl_img, pxl_lab, rm_yaxis,origin_shift ) # transform to lab coordinate

    # cleaning step 1: check coordinates out of the Z-range
    if_outlier = lab_coord[:,2] < 137 # check if the z coordinate is out of range
    lab_coord = lab_coord[if_outlier] # coord has been cleand of outliers
    lab_signal = signal[:, if_outlier] # the corresponding signal matrix has been cleaned of outliers 
    n_cells = lab_coord.shape[0] # This is cleaned of the outlyers.

    mask_labels = np.zeros([n_cells, 294])
    labels_covered = []
    labels_to_exclude = [0, 93, 274] # The large-category labels that should be excluded.
    for n_mask in range(294):
        mask_labels[:,n_mask], covered  = MD.mask_multi_direct_search(n_mask, np.fliplr(lab_coord)) # from x,y,z back to z,y,x again
        if covered:
            #mask_idx, outline_idx = MD.get_mask(n_mask)
            name_idx = MD.get_name(n_mask)
            print(name_idx)
            labels_covered.append([n_mask, covered])

    print("Finished anatomical identification.")

    # cleaning step 2: label each cell with one or more anatomical labels, remove cells that are not labeled by any mask
    mask_valid = np.sum(mask_labels, axis = 0)
    ind_mask = np.where(mask_valid)[0] # masks that contain at least one cell
    mask_labels = mask_labels[:, ind_mask]
    cells_masked = np.sum(mask_labels, axis = 1) # for each cell, sum up the masks to count how many brain regions are annotated
    ind_cell= np.where(cells_masked)[0] # cells that are tagged at least by one mask label

    if arti_clear: # do you want to remove unlabeled cells?
        mask_labels = mask_labels[ind_cell] # clean the mask labels of unmasked cells
        lab_coord = lab_coord[ind_cell] # clean the coordinates of unmasked cells
        lab_signal = lab_signal[:, ind_cell] # clean the signal of unmasked cells 

    annotation_label = np.row_stack((mask_labels, ind_mask)) # This part of information needs to be saved 

    labeled_dataset = dict()
    labeled_dataset['coord'] = lab_coord
    labeled_dataset['signal'] = lab_signal
    labeled_dataset['annotation'] = annotation_label
    np.savez(dest_path, **labeled_dataset)
    return np.array(labels_covered)


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


def coord_convert_preprocess(fpath, reg_list, origin_x,  order = 'r'):
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
        time.sleep(6) # wait until the file is saved
        f = open(dest_name, 'r')
        time.sleep(4)
        raw_coord = f.readlines()
        time.sleep(5)
        nline = len(raw_coord)
        f.close()
        fine_coord = []
        fine_data = []
        nf = 0
        # here I add some correction for FAILED coordinates.
        print(NC, nline)
        for nc in range(NC):
            s = raw_coord[nc]
            d = signal[nc]
            if 'F' in s:
                nf +=1
            elif '-' in s:
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
        return finedest_name
    else:
        print('The registration parameter does not exist.')
        return 'x'

def label_summary(group_labels, n_range = (20, 100), bar_plot = True, bar_color = 'coral'):
    '''
    summarize the anatomical labels of a fish, export a figure.
    '''
    label_file = np.load(group_labels)
    nkeys = len(label_file.keys())
    label_count = np.zeros([294, nkeys+1])
    ii = 1
    for key, sum_info in label_file.items():
        print("Fish: ", key)
        ind_labels = sum_info[:,0]
        count = sum_info[:,1]
        label_count[ind_labels, ii] = count
        ii += 1
    # now, trim the label_count matrix
    labels_valid = (label_count.sum(axis = 1) > 0) # valid labels
    label_count[:,0] = np.arange(294)
    label_count = label_count[labels_valid]
    label_mean = label_count[:,1:].mean(axis = 1)
    label_std = label_count[:,1:].std(axis = 1)
    label_covered = label_count[:,0]

    if bar_plot:
        c_min, c_max = n_range
        in_range = np.logical_and(label_mean < c_max, label_mean > c_min)
        label_plot = label_covered[in_range] # plot the labels inside the selected count range
        mean_plot = label_mean[in_range]
        std_plot = label_std[in_range]
        name_list = []
        ind = np.arange(len(label_plot))
        MD = maskdb.mask_db()
        for lab in label_plot:
            raw_name = MD.get_name(lab)
            raw_parts = raw_name.split('-')
            if len(raw_parts) > 1:
                short_name = '-'.join([raw_parts[0][:4], raw_parts[1][1:10]])
            else:
                short_name = raw_parts[0][:14]
            name_list.append(short_name)

        fig_sum = plt.figure(figsize = (8, 4.5))
        ax = fig_sum.add_subplot(111)
        ax.bar(ind, mean_plot, width = 0.8, yerr = std_plot, color = bar_color)
        ax.set_xticks(ind)
        ax.set_xticklabels(name_list, rotation = 35, ha = 'right', fontsize = 12)
        ax.set_ylabel('Counts', fontsize = 12)
        fig_sum.tight_layout()
        MD.shutdown()
        return fig_sum

    else:
        return label_mean, label_std


# ------------------------Main functions for test ----------------------
def reg_annotate():
    '''
    register and annotate
    '''
    meta_df = pd.read_csv(meta_path, sep = ',')
    meta_dim = meta_df[['Fish','NY', 'NX']]
    meta_dim.set_index('Fish', inplace = True)
    response_list = glob.glob(data_path+'FB_resting_15min/Aug2018/*_dff.npz')
    print(response_list)
    label_sum = dict()
    for response_file in response_list:
        basename ='_'.join( os.path.basename(response_file).split('.')[0].split('_')[:-1])

        print("Fish:", basename)
        xdim =dimension_check(basename, meta_dim, nyear = '18')
        print(xdim)
        temp_list = basename.split('_')
        reglist  = '_'.join([temp_list[0], temp_list[-1]])
        print(reglist)

        reg_list = data_path + 'Good_registrations/Aug2018_rest/' + reglist +  '.list'
        fine_dest_name = coord_convert_preprocess(response_file,reg_list,int(xdim),order = 'r')
        label_covered = anatomical_labeling(fine_dest_name)
        label_sum [basename] = label_covered
    label_path = data_path + 'FB_resting_15min/Jul2017_labels.npz'
    np.savez(label_path, **label_sum)

   # fig_sum = label_summary(label_path, n_range = (0,20), bar_color = 'coral')
   # fig_sum.savefig('label_summary_0_20')
   # fig_sum = label_summary(label_path, n_range = (20,100), bar_color = 'turquoise')
   # fig_sum.savefig('label_summary_20_100')
   # fig_sum = label_summary(label_path, n_range = (100,500), bar_color = 'violet')
   # fig_sum.savefig('label_summary_100_500')
   # fig_sum = label_summary(label_path, n_range = (500,5000), bar_color = 'sienna')
   # fig_sum.savefig('label_summary_500_5000')
   # # next 


def edge():

    response_list = glob.glob(data_path+'FB_resting_15min/Jul2017/*_ref.npz')
    for fname in response_list:
        edge_cropping(fname)


if __name__ == '__main__':
    reg_annotate()
