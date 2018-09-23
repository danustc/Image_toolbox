'''
A class that does spectral clustering on large datasets using the divide-and-conquer strategy.
Last update by Dan on 09/20/2018.
'''
root_path = '/home/sillycat/Programming/Python/Image_toolbox/'
import sys
sys.path.append(root_path)
import numpy as np
from spectral_clustering import Corr_sc
import matplotlib.pyplot as plt
from collections import deque
from src.visualization import signal_plot

def smart_partition(NC, n_group, last_big = True):
    arr = np.arange(NC)
    if last_big:
        g_pop = int(NC//n_group)
    else:
        g_pop = int(NC//n_group) + 1
    cutoff_pos = np.arange(n_group+1) * g_pop
    cutoff_pos[-1] = NC #reset the last element to NC
    group_index = []
    for ii in range(n_group):
        ni = cutoff_pos[ii]
        nf = cutoff_pos[ii+1]
        group_index.append(arr[ni:nf])

    return group_index

# -----------------------Below is the class of hierachical spectral clustering ------------

class hrc_sc(object):
    '''
    class description blablabla.....
    '''

    def __init__(self, raw_signal, n_group):
        self.n_group = n_group
        self.signal = raw_signal


    def divide_sc(self, threshold = 0.25, mode = 'random', interactive = True):
        '''
        spectral clustering by groups.
        partitions the whole dataset into several groups, and do spectral clustering on each group
        hard part: how to keep tracing the clustering results.
        This can be full automatic or semi-automatic.
        '''
        ncl_total = 0 # the total number of clusters
        NT, NC = self.signal.shape
        self.NT, self.NC = NT, NC
        arr = np.arange(NC)
        group_index = smart_partition(NC, self.n_group, last_big = False) # Equally partition the dataset into several groups, the last group has the smallest population.

        if mode == 'random':
            np.random.shuffle(arr)

        elif mode == 'ordered':
            pass

        self.cl_average_pool = deque() # list of lists, saving the cluster average
        self.ind_group_pool = deque() # list of lists, saving the group index average
        sc_holder = Corr_sc() # initialize an empty holder 

        for gg in range(self.n_group): # iterate over n_group
            '''
            first, evaluate the group's threshold
            '''
            sg_data = self.signal[:,group_index[gg]] # takeout a subgroup of data
            sc_holder.load_data(sg_data)
            sc_holder.link_evaluate(sca = 1.10)
            sc_holder.affinity()

            cluster_peaks, fig_plot = sc_holder.laplacian_evaluation()
            print("suggested number of clusters:", cluster_peaks)

            if interactive:
                fig_plot.show()
                n_cl = int(input("Enter the number of clusters: "))
                plt.close(fig_plot)
            else:
                if len(cluster_peaks) == 1:
                    n_cl = cluster_peaks[0]
                else:
                    n_cl = cluster_peaks[1]
            #ind_groups, cl_average = sc.spec_cluster(sg_data, n_cl)
            sc_holder.clustering(n_cl)
            self.ind_group_pool.append(sc_holder.ind_groups)
            self.cl_average_pool.append(sc_holder.cl_average)# 
            ncl_total += n_cl

        self.ncl_total = ncl_total


    def population_labeling(self):
        '''
        create an NCx2 matrix to save each neuron's group number and label number.
        First column: the group that each neuron belongs to.
        Second column: the label of each neuron within the group.
        For instance, if a cell is labeled (2,3), then it belongs to the second group and has clustering label 3.
        '''
        group_label = np.zeros((self.NC, 2))
        for ii in range(self.n_group):
            id_group, cl_aver = self.ind_group_pool[ii], self.cl_average_pool[ii]
            # next, iterate through the id_group
            n_labels = len(id_group)
            for jj in range(n_labels):
                # fill up the group label
                idx = id_group[jj]
                group_label[idx, 0] = ii
                group_label[idx, 1] = jj

        self.group_label = group_label


    def cluster_corrcheck(self):
        '''
        cross check the similarity between different clusters.
        Idea:
        1. create a similarity matrix between the different clusters.
        2. perform spectral clustering again on the corrmat
        '''
        # First, convert the cluster averages in subgroups into a big array          
        cl_average = np.column_stack(self.cl_average_pool)
        sc_holder = Corr_sc()
        sc_holder.load_data(cl_average)
        sc_holder.link_evaluate()
        sc_holder.affinity()
        cluster_peaks, fig_plot = sc_holder.laplacian_evaluation()
        fig_plot.show()
        n_cl = int(input("Enter the number of clusters: "))
        plt.close(fig_plot)
        sc_holder.clustering(n_cl)
        cluster_corrgroup = sc_holder.ind_groups
        self.trace_back(cluster_corrgroup)


    def tract_back(self, cluster_cg):
        '''
        cluster_cg: cluster y_label index, which should be traced back to self.group_label
        '''

    def merge_clusters(label_a, label_b):
        '''
        merge two clusters with label_a and label_b.
        '''

    def cluster_view(self):
        '''
        view the average of clusters
        '''
        signal_plot.compact_dffplot(fsize = (6, 4.))
