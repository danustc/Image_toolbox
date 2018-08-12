"""
A general class for analysis, which can be overloaded by many modules.
"""
import numpy as np
import sys
sys.path.append('/home/sillycat/Programming/Python/Image_toolbox')
import os
import munging

class grinder():
    '''
    the general data grinder module, which reads signal, coordinates and do shuffling when necessary.
    '''
    def __init__(self,coord, signal, rev = True):
        self.signal = signal
        self.coord = coord
        self.rev = rev # whether the coordinates are reversed.
        print("Loaded.")
        self._get_size_()

    def _get_size_(self):
        '''
        Explicitly save the data size so that we don't have to calculate the array dimensions everytime.
        '''
        self.NT, self.NC = self.signal.shape

    def reload(self, fpath, rev = True):
        data = np.load(fpath)
        self.signal = data['signal']
        self.coord = data['coord']
        self.rev = rev
        self._get_size_()

    def rev_coords(self):
        self.coord = self.coord[:,::-1]
        self.rev = not(self.rev)



    def shutDown(self):
        pass



