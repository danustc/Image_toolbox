# this is just a temporary script for testing.
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import src.dynamics.df_f as df_f

global_path = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
TS_9 = np.load(global_path+'TS_9.npz')
ts_data = TS_9['data']
dff_raw = df_f.dff_raw(ts_data, ft_width = 6, ntruncate=30)
