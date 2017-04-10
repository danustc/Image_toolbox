'''
This file loads the affine transformation output (2 triangle coordinates) and generates the affine matrix (M) and the translation vector(b).
The function trans_reading reads the triangle coordinates.
'''

import numpy as np
import numpy.linalg as linalg
import src.pipeline.stack_operations as st_op

# read affine transformation form 
def triangle2afm(ts, td):
    '''
    produce the affine matrix M and the vector d for affine transformations
    x,y: the initial coordinates
    p,q: the transformed coordinates
    '''
    LT = np.zeros(6,6)
    x1,y1 = ts[0]
    x2,y2 = ts[1]
    x3,y3 = ts[2]
    p1,q1 = td[0]
    p2,q2 = td[1]
    p3,q3 = td[2]
    LT[0] = [x1, y1, 0,0, 1.0, 0]
    LT[1] = [0, 0, x1, y1,0, 1.0]
    LT[2] = [x2, y2, 0,0, 1.0, 0]
    LT[3] = [0, 0, x2, y2,0, 1.0]
    LT[4] = [x3, y3, 0,0, 1.0, 0]
    LT[5] = [0, 0, x3, y3,0, 1.0]

    pq = np.array([p1,q1, p2, q2, p3, q3])
    paras = linalg.solve(LT, pq)
    M = paras[:4].reshape([2,2])
    b = paras[4:]
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
    for iz in np.arange(n_frames):
        line_marker = iz*10 +5  # where the coordinate number begins
        raw_d = txt_list[line_marker:(line_marker+3)]
        raw_s = txt_list[(line_marker+4):(line_marker+7)]
        tri_d = np.array([ln.split('\t') for ln in raw_d], dtype = '|S8')
        tri_s = np.array([ln.split('\t') for ln in raw_s], dtype = '|S8')
        tri_d = tri_d.astype('float64')
        tri_s = tri_s.astype('float64')

        zafm = triangle2afm(tri_s, tri_d)# tri_s, tri_d: the triangle coordinates of source and destination. 
        aff_mat.append(zafm)
