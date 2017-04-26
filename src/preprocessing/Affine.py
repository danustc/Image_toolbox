'''
This file loads the affine transformation output (2 triangle coordinates) and generates the affine matrix (M) and the translation vector(b).
The function trans_reading reads the triangle coordinates.
Last update: 04/25/2017
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox/')
import numpy as np
import numpy.linalg as linalg
import src.preprocessing.stack_operations as st_op

global_datapath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/' # this is the global path to the test data sets.
# read affine transformation form 
def triangle2afm(ts, td, mode='a'):
    '''
    produce the affine matrix M and the vector d for affine transformations
    x,y: the initial coordinates
    p,q: the transformed coordinates
    mode: a --- affine
          r --- rigid
          t --- translation
          s --- scaled rotation
    '''
    x1,y1 = ts[0]
    x2,y2 = ts[1]
    x3,y3 = ts[2]
    p1,q1 = td[0]
    p2,q2 = td[1]
    p3,q3 = td[2]

    pq = np.array([p1,q1, p2, q2, p3, q3])

    if mode == 'a':
        LT = np.zeros([6,6])
        LT[0] = [x1, y1, 0,0, 1.0, 0]
        LT[1] = [0, 0, x1, y1,0, 1.0]
        LT[2] = [x2, y2, 0,0, 1.0, 0]
        LT[3] = [0, 0, x2, y2,0, 1.0]
        LT[4] = [x3, y3, 0,0, 1.0, 0]
        LT[5] = [0, 0, x3, y3,0, 1.0]
        paras = linalg.solve(LT, pq)
        M = paras[:4].reshape([2,2])
        b = paras[4:]
    elif mode == 'r':
        # rigid body transformation, less degrees of freedom
        LT = np.zeros([4,4])
        LT[0] = [x1, -y1, 1.0, 0]
        LT[1] = [y1, x1, 0, 1.0]
        LT[2] = [x2, -y2, 1.0, 0]
        LT[3] = [y2, x2, 0, 1.0]
        paras = linalg.solve(LT,pq[:4])
        u = paras[0]
        v = paras[1]
        M = np.array([[u,-v], [v,u]])
        b = paras[2:]
    return M, b

def aff_read(tm_path):
    '''
    The transformation file has the standard format of the .txt output file produced by MultiStackReg (Brad Busse).
    '''
    f = open(tm_path, 'r')
    txt_contents = f.read()
    f.close()
    txt_list = txt_contents.splitlines()
    transform_method = txt_list[3] # it can be 'TRANSLATION', 'RIGID_BODY', 'AFFINE', and this method string serves as the landmark of the output file. 
    n_frames = txt_list.count(transform_method)
    aff_mat = [] # affine matrix
    aff_vec = []
    for iz in np.arange(n_frames):
        line_marker = iz*10 +5  # where the coordinate number begins
        raw_d = txt_list[line_marker:(line_marker+3)]
        raw_s = txt_list[(line_marker+4):(line_marker+7)]
        tri_d = np.array([ln.split('\t') for ln in raw_d], dtype = '|S8')
        tri_s = np.array([ln.split('\t') for ln in raw_s], dtype = '|S8')
        tri_d = tri_d.astype('float64')
        tri_s = tri_s.astype('float64')

        zafm, zafb = triangle2afm(tri_s, tri_d, mode = 'r')# tri_s, tri_d: the triangle coordinates of source and destination. 
        aff_mat.append(zafm)
        aff_vec.append(zafb)


    return aff_mat, aff_vec

# next, apply the affine transformation to the frame (this is a must-test)
def aff_transform(frame, afm, afb):
    '''
    frame: the original image frame
    afm: affine transformation matrix
    afb: affine transformation shift vector
    That also involves interpolation of the transformed image
    '''
    tframe = np.zeros_like(frame)

    return tframe

def pixel_transform(px_list, afmat, afvec):
    '''
    afmat: the matrix part of the affine transformation
    afvec: the vector part of the affine transformation
    px_list: the list of pixels to be transformed. Conventions: column 0 -- y; column 1 -- x. Need to be transposed before applying the affine matrix on it.
    '''
    if(px_list.size == 2):
        # if px list is only one point
        px_trans = afmat*px_list+afvec
    else:
        NL = np.max(px_trans.shape)
        px_trans_row = np.dot(afmat, px_list.transpose()) +np.tile(afvec, (NL,1).transpose())
        px_trans = px_trans_row.transpose()
    return px_trans

# -----------------------------------------_Testing function (main)--------------------------------------

def main():
    '''
    test for functions
    '''
    tmpath_ts2zd = global_datapath+'sliceReg.txt'
    rigmat_multi, rigvec_multi = aff_read(tmpath_ts2zd)
    print("Number of aligned slices:", len(rigmat_multi))
    print(rigmat_multi)
    print(rigvec_multi)





if __name__ == '__main__':
    main()
