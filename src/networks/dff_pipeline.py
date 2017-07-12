'''
Created by Dan on 07/11/2017.
Pipeline for batch calculation of raw_f into Delta F/F.
'''
import os
import glob
import numpy as np
from df_f import *


global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/HQ/'
# ---------------------Below are small functions for data cleaning -------------

def raw2dff_clean(raw_fname, dff_flag = 'dff', dt = 0.5, t_width = 1.5, saveraw = False):
    '''
    convert a raw fluorescence file into DF/F values.
    '''
    f_dic = dict(np.load(raw_fname)) # convert the Npz file into dictionary
    data_raw = f_dic.pop('data')
    dffr = dff_raw(data_raw, ft_width = 6, ntruncate = 20)
    if saveraw:
        f_dic['dff_raw'] = dffr
    f_dic['signal'] = dff_expfilt_group(dffr, dt, t_width)
    f_dic['t_features'] = np.array([dt, t_width]) # save the temporal information
    folder_path = os.path.dirname(raw_fname) +'/'
    dff_fname = os.path.basename(raw_fname).split('.')[0]+ '_' + dff_flag
    np.savez(folder_path + dff_fname, **f_dic)

#------------------------------The main test function ---------------------

def main():
    folder_list = glob.glob(global_datapath+'*')
    for folder in folder_list:
        raw_fname = folder + '/merged.npz'
        raw2dff_clean(raw_fname, saveraw = False)
        print("Finished processing:", folder.split('/')[-1])


if __name__ == '__main__':
    main()
