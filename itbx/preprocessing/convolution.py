'''
This is the Data generation script.
'''
import numpy as np
from itbx.preprocessing.correlation import fft_image, ift_image

def conv_2D(image, psf, padding = 10, mode = 'center', pyfftw = True):
    '''
    image: the ground truth image
    padding:pad each dimension by 10 pixels.
    '''
    NY, NX = image.shape
    PY, PX = psf.shape # PSF dimensions 
    DY, DX = NY-PY, NX-PX
    HY, HX = int(DY//2), int(DX//2)

    # padding the PSF and the image 
    padded_PSF = np.zeros((NY+2*padding, NX+2*padding))
    padded_PSF[HY+padding:HY+padding+PY, HX+padding:HX+padding+PX] = psf # created the padded psf

    if padding > 0:
        padded_image = np.pad(image, ((padding,padding),(padding,padding)), mode = 'constant')
    else:
        padded_image = image
    ft_image = fft_image(padded_image, abs_only = False, shift = False, use_pyfftw = pyfftw)
    ft_psf = fft_image(padded_PSF, abs_only = False, shift = False, use_pyfftw = pyfftw)


    conv_stack = ift_image(ft_stack*ft_psf, abs_only = True, shift_back = False, use_pyfftw = pyfftw)
    return conv_stack


def conv_3D(stack, psf, padding_xy = 10, padding_z = 1,PYFFTW = True):
    '''
    stack: the ground truth image stack
    psf: the 3D PSF.
    '''
    NZ, NY, NX = stack.shape
    PZ, PY, PX = psf.shape
    DZ, DY, DX = NZ-PZ, NY-PY, NX-PX
    HZ, HY, HX = int(DZ//2), int(DY//2), int(DX//2)

    padded_psf = np.zeros((NZ+2*padding_z, NY+2*padding_xy, NX+2*padding_xy))
    padded_psf[HZ+padding_z:HZ+padding_z+PZ, HY+padding_xy:HY+padding_xy+PY, HX+padding_xy:HX+padding_xy+PX] = psf # create a padded psf.

    padded_stack = np.pad(stack,((padding_z, padding_z),(padding_xy, padding_xy), (padding_xy, padding_xy) ), mode = 'constant')

    ft_stack = fft_image(padded_stack,abs_only = False, shift = False, use_pyfftw=PYFFTW)
    ft_psf = fft_image(padded_psf, abs_only = False, shift = False, use_pyfftw = PYFFTW)

    print("Performed fft.")
    conv_stack = ift_image(ft_stack*ft_psf, abs_only = True, shift_back = False, use_pyfftw=PYFFTW)
    print("performed ift.")
    return conv_stack


