import os
import sys
import glob
import numpy as np
import subprocess
import time

exu = "cmtk streamxform"
data_path="/home/sillycat/Programming/Python/data_test/"


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


def coord_convert_preprocess(fpath, reg_list):
    raw_data = np.load(fpath)
    basename = os.path.basename(fpath).split('.')[0].split('_')[0]
    parent_path= os.path.dirname(fpath)
    coord = raw_data['coord']
    coord[:,2] = 975*0.295-coord[:,2]
    try:
        signal = raw_data['data'].T
    except KeyError:
        try:
            signal = raw_data['signal'].T
        except KeyError:
            return -1
    source_name = parent_path + '/' + basename + '_coord.txt'
    dest_name = parent_path + '/' + basename + '_rawref.txt'
    finedest_name = parent_path + '/' + basename + '_ref'
    NC = coord.shape[0]
    if os.path.isdir(reg_list):
        np.savetxt(source_name, coord[:,::-1], fmt = '%12.5f')
        registered_coord_convert(reg_list, source_name, dest_name, inv = True)
        time.sleep(5) # wait until the file is saved
        f = open(dest_name, 'r')
        raw_coord = f.readlines()
        f.close()
        fine_coord = []
        fine_data = []
        nf = 0
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


def main():
    response_list = glob.glob(data_path+'Good_registrations/*merged.npz')
    print(response_list)
    for response_file in response_list:
        basename = os.path.basename(response_file).split('.')[0].split('_')[0]
        reg_list = data_path + 'Good_registrations/' + basename + '.list'
        sta = coord_convert_preprocess(response_file,reg_list)
        print(sta)


if __name__ == '__main__':
    main()
