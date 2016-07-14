from Preprocess import Preprocess
import matplotlib.pyplot as plt
import tifffunc
import trackpy as tp
import numpy as np
from skimage.filters import threshold_adaptive
from skimage import color, io, filters

plt.close('all')  

impp = Preprocess('test_image')
new_stack = impp.stack_high_trunc()

# image = tifffunc.read_tiff('test_image')
# edges = filters.sobel(image[0])
# plt.imshow(edges)
# plt.show()
# fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(12, 6))


# ax0, ax1 = axes.flat


# f1 = tp.locate(image[5], 11, minmass = 1000, invert = False, threshold=100)
# f2 = tp.locate(image[5], 15, minmass = 1000, invert = False, threshold=100)



# ax0.imshow(dilated,cmap = 'Greys_r')
# ax1.imshow(gs1, cmap = 'Greys_r')

tifffunc.write_tiff(new_stack, 'new_stack_s4')
# tp.annotate(f2, image[5], ax=ax0)s
# tp.annotate(f4, gs1, ax=ax1)

