'''
A class that does spectral clustering on large datasets using the divide-and-conquer strategy.
Last update by Dan on 09/20/2018.
'''

import numpy as np
import spectral_clustering as sc
import matplotlib.pyplot as plt
import collections import deque

class hrc_sc(object):
    '''
    class description blablabla.....
    '''

    def __init__(self, raw_signal, n_group):
        self.n_group = n_group
        self.signal = signal


    def divide_sc(self, threshold = 0.25, mode = 'random', interactive = True):
        '''
        spectral clustering by groups.
        partitions the whole dataset into several groups, and do spectral clustering on each group
        hard part: how to keep tracing the clustering results.
        This can be full automatic or semi-automatic.
        '''
        ncl_total = 0 # the total number of clusters
        NT, NC = self.signal.shape
        arr = np.arange(NC)
        group_index = sc.smart_partition(NC, self.n_group, last_big = False) # Equally partition the dataset into several groups, the last group has the smallest population.

        if mode == 'random':
            np.random.shuffle(arr)

        elif mode == 'ordered':
            pass

        self.cl_average_pool = deque() # list of lists, saving the cluster average
        self.ind_group_pool = deque() # list of lists, saving the group index average
        for gg in range(self.n_group): # iterate over n_group
            '''
            first, evaluate the group's threshold
            '''
            sg_data = self.signal[:,group_index[gg]] # takeout a subgroup of data
            cluster_peaks, th = dataset_evaluation(sg_data)
            print("suggested number of clusters:", cluster_peaks)
            print("suggested threshold:", th)

            if interactive:
                n_cl = int(input("Enter the number of clusters: "))
            else:
                if len(cluster_peaks) == 1:
                    n_cl = peak_position[0]
                else:
                    n_cl = peak_position[1]
            ind_groups, cl_average = spec_cluster(sg_data, n_cl)
            self.ind_group_pool.append(ind_groups)
            self.cl_average_pool.append(cl_average)# 
            ncl_total += n_cl

        self.ncl_total = ncl_total


    def labeling_assignment(self):
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
                idx = id_group[jj]
                group_label[idx] = ii
                group_label[id_group,1] = 



    def cluster_crosscheck(self):
        '''
        cross check the similarity between different clusters.
        Idea:
        create a similarity matrix between the different clusters.
        '''
        
