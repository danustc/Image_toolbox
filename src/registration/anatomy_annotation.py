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
    command = exu + ' -- --inverse %s < %s > %s'
    #subprocess.call([self.executable, files_file, str(self.multiplier),
    subprocess.Popen(command%(rg_path, coord_source, coord_dest), shell = True)



def main():
    response_list = glob.glob(data_path+'Good_registrations/*resp.npz')
    print(response_list)
    for response_file in response_list:
        '''
        check the filename
        '''
        basename = os.path.basename(response_file).split('.')[0].split('_')[0]
        print(basename)
        resp = np.load(response_file)
        coord = resp['coord']
        source_name = data_path + 'Good_registrations/' + basename + '_coord.txt'
        dest_name = data_path + 'Good_registrations/' + basename + '_rawref.txt'
        finedest_name = data_path + 'Good_registrations/' + basename + '_ref.txt'
        reg_list = data_path + 'Good_registrations/' + basename + '.list'
        if os.path.isdir(reg_list):
            np.savetxt(source_name, coord, fmt = '%12.5f')
            registered_coord_convert(reg_list, source_name, dest_name)
            time.sleep(1)
            f = open(dest_name, 'r')
            raw_coord = f.readlines()
            f.close()
            fine_coord = []
            for s in raw_coord:
                if 'F' in s:
                    pass
                else:
                    fine_coord.append(s)
            of = open(finedest_name, 'a+')
            of.writelines(fine_coord)
            print(fine_coord)
        else:
            print("The registration result does not exist.")


if __name__ == '__main__':
    main()
