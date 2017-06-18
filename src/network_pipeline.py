'''
A small pipeline for network analysis. Added by Dan on 06/18/2017.
Last update: 06/18/2017.
'''
import sys
sys.path.append('/home/fillycat/Programming/Image_toolbox')
import numpy as np
import src.dynamics.df_f as df_f # the functions of calculating dff
import src.visualization.stat_present as stat_present
import src.networks.pca_sorting as pca_sorting
import src.networks.ica_sorting as ica_sorting

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'

class pipeline(object):
    '''
    Can I follow the design of inControl, name all the core data processing classes as pipeline?
    Purpose:    0. Load all the .npz files in a folder. These should all be T-slices
    '''
    def __init__(self, work_folder, fname_flag ='TS'):
        data_list = glob.glob(work_folder+ '*'+ fname_flag+'*.npz')
        if len(data_list) == 0:
            print("Error! There are no data file in the selected folder.")
        else:
            self.data_list = data_list
            self._load_data_()

    def _load_data_(self,verbose = True):
        for fname in self.data_list:
            fname_stem = os.path.split





def main():
    '''
    The test function of the pipeline.
    '''
    raw_fname = global_datapath+'Jun13_A1_GCDA/'
    dff_raw, f_base = df_f.dff_raw(Dec07_B1_data, ft_width=4, ntruncate = 20)
