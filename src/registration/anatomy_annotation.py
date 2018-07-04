import os
import sys
import glob
import numpy as np
import subprocess
import time
import pandas as pd
import calendar

exu = "cmtk streamxform"
data_path="/home/sillycat/Programming/Python/data_test/"
meta_path = data_path + 'metadata_sheet.csv'
month_num = {v: k for k,v in enumerate(calendar.month_abbr)}
print(month_num)



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
