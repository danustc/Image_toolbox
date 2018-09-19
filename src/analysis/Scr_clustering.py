'''
Sample script: clustering of a large group of neurons
'''
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
from src.analysis.Analysis import grinder
import matplotlib.pyplot as plt


global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/Jul2017/'
portable_datapath = '/media/sillycat/DanData/'

def main():
    grinder_core = grinder()
    grinder_core.parse_data(data_path)
    grinder_core.activity_sorting()
    grinder_core.background_suppress(sup_coef = 0.0001)
