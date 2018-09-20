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
        self.signal = signal


    def divide_sc(self, n_group, threshold = 0.25, mode = 'random', interactive = True):
        '''
        spectral clustering by groups.
        partitions the whole dataset into several groups, and do spectral clustering on each group
        hard part: how to keep tracing the clustering results.
        This can be full automatic or semi-automatic.
        '''
        NT, NC = self.signal.shape
        arr = np.arange(NC)
        group_index = sc.smart_partition(NC, n_group, last_big = False) # Equally partition the dataset into several groups, the last group has the smallest population.

        if mode == 'random':
            np.random.shuffle(arr)

        elif mode == 'ordered':
            pass

        self.cl_average_pool = deque() # list of lists, saving the cluster average
        self.ind_group_pool = deque() # list of lists, saving the group index average
        for gg in range(n_group): # iterate over n_group
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
            self.cl_average_pool.append(cl_average)



    def cluster_crosscheck(self):
        '''
        cross check the similarity between different clusters.
        Idea:
        create a similarity matrix between the different clusters.
        '''
