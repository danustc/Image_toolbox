'''
Convert a series of coordinates (x,y,z in microns) into txt list file.
'''
import os
import numpy as np
import glob as glob
global_datapath = '/home/sillycat/Programming/Python/data_test/'

def coord_convert(coord, dest_path, reorder = None):
    '''
    convert z,y,x coordinates into x,y,z
    '''

    if reorder is not None:
        coord = coord[:, reorder]
    np.savetxt(dest_path, coord, fmt = '%.4f')


def coord_integrate(coord_new, dataset, dest_path):
    new_dataset = dict()
    new_dataset['coord'] = coord_new
    new_dataset['signal'] = dataset['signal']
    np.savez(dest_path, **new_dataset)


def main():
    data_list = glob.glob(global_datapath + 'Jun_GCDA/'+'Jun27_B1*cl*.txt')
    for df_name in data_list:
        coord_new = np.loadtxt(df_name)
        #coord_new[:,1] = 300-coord_new[:,1]
        df_base = os.path.basename(df_name).split('.')[0]
        cluster_base = df_base+'.npz'
        df_folder = os.path.dirname(df_name)
        cluster_set = np.load(df_folder+'/'+cluster_base)
        coord_convert(cluster_set['coord'], df_folder + '/'+ df_base + '_coord.list')
        coord_integrate(coord_new, cluster_set, df_folder+'/'+ df_base + '_reform')


if __name__ == '__main__':
    main()

