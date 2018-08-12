"""
A general class for analysis, which can be overloaded by many modules.
"""
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os


class Pipeline:
    '''
    The general analysis module
    '''
    def __init__(self,coord, signal):
        self.signal = signal
        self.coord = coord
        print("Loaded.")


    def shutDown(self):
        pass



