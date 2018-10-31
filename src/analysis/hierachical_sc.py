'''
A class that does spectral clustering on large datasets using the divide-and-conquer strategy.
Last update by Dan on 09/20/2018.
'''
root_path = '/home/sillycat/Programming/Python/Image_toolbox/'
import sys
sys.path.append(root_path)
import numpy as np
from spectral_clustering import Corr_sc, label_assignment
import matplotlib.pyplot as plt
from collections import deque
from src.visualization import cluster_navigation

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


    def divide_sc(self, mode = 'random', interactive = False):
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
        self.group_partition = deque()
        self.group_cstat = np.zeros((self.n_group, 2)) # group cluster number and threshold

        sc_holder = Corr_sc() # initialize an empty holder 

        for gg in range(self.n_group): # iterate over n_group
            '''
            first, evaluate the group's threshold
            '''
            neuron_ind = arr[group_index[gg]] # if arr is not shuffled, then it is the same as group_index[gg]
            self.group_partition.append(neuron_ind)
            sg_data = self.signal[:,neuron_ind] # takeout a subgroup of data
            sc_holder.load_data(sg_data)
            sc_holder.link_evaluate(sca = 1.050)
            sc_holder.affinity()
            self.group_cstat[gg, 0] = sc_holder.thresh

            cluster_peak, fig_plot = sc_holder.laplacian_evaluation()
            print("suggested number of clusters:", cluster_peak)

            fig_plot.show()

            if interactive:
                n_cl = int(input("Enter the number of clusters: "))
            else:
                # figure the position of peak
                n_cl = cluster_peak
                plt.pause(1)

            plt.close(fig_plot)
            self.group_cstat[gg,1] = n_cl
            sc_holder.clustering(n_cl)
            self.ind_group_pool.append(sc_holder.ind_groups)
            self.cl_average_pool.append(sc_holder.cl_average)# 
            ncl_total += n_cl

        self.ncl_total = ncl_total


    def group_result_display(self, t_pause = 3):
        '''
        display group result temporarily, pause for t_pause seconds and turn off. No saving.
        '''



    def groupwise_population_labeling(self):
        '''
        create an NCx2 matrix to save each neuron's group number and label number.
        First column: the group that each neuron belongs to.
        Second column: the label of each neuron within the group.
        For instance, if a cell is labeled (2,3), then it belongs to the second group and has clustering label 3.
        '''
        neuron_label = np.zeros((self.NC, 2))
        cluster_label = np.zeros((self.ncl_total,2))
        cl_start = 0
        for ii in range(self.n_group):
            id_group, cl_aver = self.ind_group_pool[ii], self.cl_average_pool[ii]
            #id_group, cl_aver = self.group_partition[ii], self.cl_average_pool[ii]
            id_partition = self.group_partition[ii]
            # next, iterate through the id_group
            n_labels = len(id_group)
            cluster_label[cl_start : cl_start+n_labels, 0] = ii
            cluster_label[cl_start : cl_start+n_labels, 1] = np.arange(n_labels)
            for jj in range(n_labels):
                # fill up the group label
                # check the neuron
                idx = id_partition[id_group[jj]]
                neuron_label[idx, 0] = ii
                neuron_label[idx, 1] = jj
            cl_start += n_labels

        self.neuron_label = neuron_label
        self.cluster_label = cluster_label



    def cluster_corrcheck(self):
        '''
        cross check the similarity between different clusters.
        Idea:
        1. create a similarity matrix between the different clusters.
        2. perform spectral clustering again on the corrmat
        Note on 09/24: I should be cautious merging clusters between subgroups
        '''
        # First, convert the cluster averages in subgroups into a big array          
        print("Perform cross check of the clusters:")
        cl_average = np.column_stack(self.cl_average_pool)
        sc_holder = Corr_sc()
        sc_holder.load_data(cl_average)
        cmat = sc_holder.corr_mat
        plt.imshow(cmat)
        plt.show()
        plt.pause(1)
        plt.close()
        sc_holder.link_evaluate(sca = 1.80)
        sc_holder.affinity()
        cluster_peak, fig_plot = sc_holder.laplacian_evaluation(ncl = 30)
        fig_plot.show()
        self.n_supgroup = cluster_peak
        plt.pause(1)
        plt.close(fig_plot)
        sc_holder.clustering(self.n_supgroup)
        cluster_corrgroup = sc_holder.ind_groups
        print(cluster_corrgroup)
        return cluster_corrgroup


    def merge_clusters(self, cluster_cg):
        '''
        cluster_cg: cluster y_label index, which should be traced back to self.group_label
        for each cluster, trace back its i,j
        '''
        merged_label = np.zeros(self.NC)
        print("# of super groups:", self.n_supgroup)
        cell_ind = []
        for ii in range(self.n_supgroup):
            ind_cg =  cluster_cg[ii]
            label_cg = self.cluster_label[ind_cg]
            merged_group = []
            for labels in label_cg:
                '''
                first element: the group number
                second element: the label number within that group
                '''
                search_neurons = np.where(np.logical_and(self.neuron_label[:,0]==labels[0], self.neuron_label[:,1]==labels[1]))[0]
                merged_group.append(search_neurons)
            # After iterating through the labels
            ind_mg = np.concatenate(merged_group) # indices of the merged group
            if len(ind_mg)==0:
                print("Error! ")
            merged_label[ind_mg] = ii
            cell_ind.append(ind_mg)

        ind_supgroups, cl_supaverage = label_assignment(self.signal, self.n_supgroup, merged_label)

        # This is solely for checking the group populations
        return merged_label, cl_supaverage, ind_supgroups

