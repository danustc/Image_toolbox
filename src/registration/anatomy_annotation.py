import os
import sys

import subprocess

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
    good_reglist = data_path+ 'Good_registrations/RFP_Jun27B1GCDA_warp_m0g160c8e1e-1x52r2.list'
    source = data_path + 'Jun_GCDA/Jun27_B1_merged_dff_cl_3_coord.list'
    dest = data_path + 'test_cl3.txt'
    registered_coord_convert(good_reglist, source, dest)

if __name__ == '__main__':
    main()
