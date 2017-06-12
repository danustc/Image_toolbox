'''
Created by Dan on 05/19/2017. For imaging display.
'''
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import numpy as np
import matplotlib.pyplot as plt
import src.preprocessing.tifffunc as tifffunc

def cmap_group(cm_list, n_group, axis = 0):
    '''
    cm_list: a list of cmap labels
    n_group: the number of groups
    axis: 0 --- by row (each row share the same color coding)
          1 --- by column
    '''
    tcm = np.tile(np.array(cm_list), (n_group,1))
    if axis == 0:
        return tcm.T
    else:
        return tcm



def superplot(im_array, nrow, ncol, as_ratio = 1., cmaps = 'Greys_r', cm_axis = 0, row_labels = [] , col_labels = [], padding = [0.05, 0.05]):
    '''
    im_array: an array of images, all in 2D arrays, can be different in size. Assume that the order of the array is arranged by rows, i.e., I[0][0], I[0][1] ... I[0][N]; I[1][0], I[1][1], ... I[1][N]; ... I[M][N].
    nrow: the number of rows in the subplot array
    ncol: the number of columns in the subplot array
    as_ratio: aspect ratio of each subplot, width/height
    cmaps: a list of cmaps. If the number of cmaps matches the number of subplots, then each subplot has an individual color map
    cm_axis: 0 --- The rows share the same color map
                1 --- The columns share the same color map
                None --- each subplot is coded independently.
    ax_num: whether the axis numbers should be shown or not.
    row_labels: Add the row labels on the very left
    col_labels: Add the col_labels on top of the plots
    padding: the w space and h space between subfigures
    '''


    nplots = len(im_array)
    if(nrow * ncol !=nplots):
        print("The grid array and the image array size mismatch! ")
        return None
    else:
        fw = (2.0+padding[0])*ncol
        fh = (2.0/as_ratio+padding[1])*nrow # the figure height
        fig_pc, axes = plt.subplots(nrow, ncol)
        if isinstance(cmaps, list): # if cmaps is a list
            ngroup = [ncol, nrow][cm_axis]
            print(cmaps, ngroup, cm_axis)
            cm_group = cmap_group(cmaps, ngroup, cm_axis)
        else: # if cmaps is a string
            cm_group = np.empty((nrow, ncol), dtype = '<U10') # create an array of empty strings
            cm_group[:] = cmaps # Now we can use the uniform cmaps for all the subplots.

        fig_pc.subplots_adjust(hspace = padding[0], wspace = padding[1]) # change the padding

        iscol_label = (len(col_labels) == ncol) # whether to label columns and rows
        isrow_label = (len(row_labels) == nrow)
        if(iscol_label):
            # increase the figure height by 0.02
            fh+=0.2
        if(isrow_label):
            fw+=0.05
        fig_pc.set_size_inches(fw,fh)


        for nr in range(nrow):
            for nc in range(ncol): # iterate through columns
                ax = axes[nr, nc]
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.imshow(im_array[nr*ncol+nc], cmap = cm_group[nr,nc])
                if ax.is_first_col() and isrow_label:
                    ax.yaxis.set_visible(True)
                    ax.set_yticks([])
                    ax.set_ylabel(row_labels[nr], fontsize = 12)
                if ax.is_first_row() and iscol_label:
                    ax.set_title(col_labels[nc], fontsize = 12)

    plt.tight_layout()
    return fig_pc



# -----------------------Main function for testing -------------------

def main():
    impath = '/home/sillycat/Programming/Python/Image_toolbox/data_test/'
    test_stack = tifffunc.read_tiff(impath+'Oct25_B3_TS18.tif', nslice = np.arange(12))
    print(test_stack.shape)
    fig_pc = superplot(test_stack, 3, 4, 1.172, ['Greens_r', 'Reds_r', 'Greys_r'], 0, row_labels = ['group 1', 'group 2', 'group 3'], col_labels = ['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4'], padding = [0.02, 0.02])
    fig_pc.savefig(impath+'super_test')


if __name__ == '__main__':
    main()
