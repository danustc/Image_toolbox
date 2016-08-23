"""
Added by Dan on 08/22/2016. To deal with redundancy. 
Can be run independently. Operated on the key-value data instead of the raw images.
After redundancy removal, the entire z-y-x-t can be saved as a 3-D npy array.
"""

import numpy as np
import glob 


class redundant(object):
    def __init__(self, work_folder):
        """
        work_folder: the folder containing all the npz files.
        
        """
        self.work_folder = work_folder
    
    def __z_construct__(self):
        """ 
        construct a virtual z-stack, do the alignment 
        """
        

    


    def red_detect(self):
        """
        Detect redundancy 
        """ 
        
        
        