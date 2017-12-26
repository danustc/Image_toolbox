from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import deformable_registration
import numpy as np
import time
import glob

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/FB_resting_15min/'
ref_datapath = '/home/sillycat/Programming/Python/Image_toolbox/cmtkRegistration/CPD/'

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue')
    plt.draw()
    print("iteration %d, error %.5f" % (iteration, error))
    plt.pause(0.001)

def main():
    # load the reference data
    data_list = glob.glob(global_datapath + '*merged_dff.npz')

    x_ref = np.load(ref_datapath+'compressed.npy')
    x_ref[:,0]*=8.0
    x_ref[:,1:]*=0.406 # the pixel size of the reference template
    y_gmm = np.load(data_list[0])['coord']


    reg = deformable_registration(x_ref, y_gmm)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = 90., azim = 90.)
    callback = partial(visualize, ax=ax)
    TY = reg.register(callback)[0]

    fig.savefig(global_datapath+ 'CPD_test_view_z')
    ax.view_init(elev = 60., azim = 90.)
    fig.savefig(global_datapath + 'CPD_test_view_a')
    ax.view_init(elev = 60., azim = 240.)
    fig.savefig(global_datapath + 'CPD_test_view_b')
if __name__ == '__main__':
    main()
