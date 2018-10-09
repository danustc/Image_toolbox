import numpy as np
import pyfftw


def rotmat_yaxis(ra):
    '''
    Calculate the rotational matrix with respect to y.
    ra: rotational angle in degrees
    '''
    rotmat = np.zeros((3,3))
    rad_ra = ra*np.pi/180.0
    rotmat[0] = np.array([np.cos(rad_ra), 0, np.sin(rad_ra)])
    rotmat[1,1] = 1.0
    rotmat[2] = np.array([-np.sin(rad_ra), 0, np.cos(rad_ra)])
    return rotmat

def trilinear_interpolation(stack, coord_px):
    '''
    trilinear interpolation in the 3d space.coord_px: x,y,z coordinates in the unit of pixel. Need to be permuted?
    if the coord_px exceeded the stack size, assign 0.0
    '''
    nz, ny, nx = stack.shape
    cx = coord_px[0]
    cy = coord_px[1]
    cz = coord_px[2]
    x0 = int(np.floor(cx))
    y0 = int(np.floor(cy))
    z0 = int(np.floor(cz))
    xd = cx-x0
    yd = cy-y0
    zd = cz-z0

    x1 = x0+1
    y1 = y0+1
    z1 = z0+1

    try:
        p000 = stack[z0, y0, x0]
    except IndexError:
        p000 = 0.
    try:
        p100 = stack[z0, y0, x1]
    except IndexError:
        p100 = 0.
    try:
        p010 = stack[z0, y1, x0]
    except IndexError:
        p010 = 0.
    try:
        p001 = stack[z1, y0, x0]
    except IndexError:
        p001 = 0.
    try:
        p110 = stack[z0, y1, x1]
    except IndexError:
        p110 = 0.
    try:
        p101 = stack[z1, y0, x1]
    except IndexError:
        p101 = 0.
    try:
        p011 = stack[z1, y1, x0]
    except IndexError:
        p011 = 0.
    try:
        p111 = stack[z1, y1, x1]
    except IndexError:
        p111 = 0.

    dx_y0z0 = xd*p100+(1.-xd)*p000
    dx_y1z0 = xd*p110+(1.-xd)*p010
    dx_y0z1 = xd*p101+(1.-xd)*p001
    dx_y1z1 = xd*p111+(1.-xd)*p011

    dy_z0 = yd*dx_y1z0 + (1.-yd)*dx_y0z0
    dy_z1 = yd*dx_y1z1 + (1.-yd)*dx_y0z1

    dz = zd*dy_z1 + (1.-zd)*dy_z0

    return dz

def sample_from_refstack(stack_ref, sample_range, pxl_ref, pxl_sample,  rotmat, rshift):
    '''
    stack_ref: the reference stack, i.e., the Z-brain.
    sample_range: the range of sampling, in pixels.
    pxl_ref: the pixel size of the reference stack
    pxl_sample: the pixel size of the sample stack.
    r_shift: the translation vector from the lab frame to the image frame, unit of pixel.
    Warning: the origin of the image stack is the geometric center of the stack, i.e., (hx, hy, hz); the origin of the reference stack is (0,0,0). Unit in pixels.
    '''
    sx, sy, sz = sample_range

    hx = sx//2
    hy = sy//2
    hz = sz//2
    print("Half range:", hx, hy, hz)
    ix = (np.arange(sx)-hx)*pxl_sample[0]
    iy = (np.arange(sy)-hy)*pxl_sample[1]
    iz = (np.arange(sz)-hz)*pxl_sample[2]
    [MY, MZ, MX]=np.meshgrid(iy, iz, ix) # the grid coordinates of the image stacks
    fmz = np.ndarray.flatten(MZ)
    fmy = np.ndarray.flatten(MY)
    fmx = np.ndarray.flatten(MX)
    flat_imgcoord = np.c_[fmx, fmy, fmz]
    flat_labcoord = (np.dot(flat_imgcoord, rotmat))/pxl_ref+rshift # the coordinates in the lab frame in the unit of pixels. 
    inter_signal = np.empty(sx*sy*sz) # initialize an empty array
    ii = 0
    for coord in flat_labcoord:
        inter_signal[ii] = trilinear_interpolation(stack_ref, coord)
        ii+=1
        if ii%10000 == 0:
            print("----------Finished %d out of %d calculations.------------"%(ii,sx*sy*sz))

    sample_value = np.reshape(inter_signal, [sz, sy, sx]) #note that the order of indices in 3D python array.
    return sample_value.astype('uint16')


def sample_to_refstack_list(coord_list, sample_range, pxl_sample, pxl_ref, rotmat, rshift):
    '''
    convert a list of coordinates into the coordinates in the reference frame.
    '''
    half_range = sample_range//2
    img_coord = (coord_list/pxl_sample-half_range)*pxl_sample
    lab_coord = np.dot(img_coord, rotmat)/pxl_ref + rshift

    return lab_coord



def sample_to_refstack(substack, ref_range, pxl_sample, pxl_ref, rotmat, rshift):
    '''
    A reference stack with size known, a registered chunk of image, which should be filled into the empty reference frame.
    '''
    ref_stack = np.zeros(ref_range) # create an empty stack
    rz, ry, rx = ref_range
    sz, sy, sx = substack.shape
    hz = sz//2
    hx = sx//2
    hy = sy//2
    idx_offset = np.array([hx,hy,hz])
    ix = (np.arange(sx)-hx)*pxl_sample[0]
    iy = (np.arange(sy)-hy)*pxl_sample[1]
    iz = (np.arange(sz)-hz)*pxl_sample[2]
    [MY, MZ, MX]=np.meshgrid(iy, iz, ix) # the grid coordinates of the image stacks
    fmz = np.ndarray.flatten(MZ)
    fmy = np.ndarray.flatten(MY)
    fmx = np.ndarray.flatten(MX)
    flat_imgcoord = np.c_[fmx, fmy, fmz]
    flat_labcoord = np.round((np.dot(flat_imgcoord, rotmat))/pxl_ref+rshift) # the coordinates in the lab system, unit pixel
    flat_imgcoord_approx = np.dot((flat_labcoord-rshift)*pxl_ref, rotmat.T)/pxl_sample   # convert back to the image coordinate
    ii = 0
    for coord in flat_imgcoord_approx:
        indx = flat_labcoord[ii,[2,1,0]].astype('int') # nz, ny, nx
        img_coord = coord+idx_offset
        if (indx[0] < rz and indx[1] < ry and indx[2] < rx):
            if(sx > img_coord[0] and sy > img_coord[1] and sz > img_coord[2]):
                ref_stack[indx[0], indx[1], indx[2]] = trilinear_interpolation(substack, img_coord)
        ii+=1
        if ii%10000 == 0:
            print("--------Finished %d out of %d calculations. ------"%(ii, sx*sy*sz))
    return ref_stack.astype('uint16')
    # [MY, MZ, MX] = np.meshgrid(iy,iz, ix)


# --------------------------------------------------Below are functions for two-photon ----------------------------------------------- 
def crop_recover(coords, offset, ref_range):
    '''
    recover the coordinates of images
    crops: the cropping corner,  ordered in z-y-x, 3x2 array.
    coords: the z-y-x coordinates
    Everything in the unit of pixel.
    '''
    offset = crops[:,0]-1 # the real offset. The cropping starts from 1 but the array starts from 0.

    ref_coords = coords+offset

    return ref_coords

