"""
Created by Dan Xie on 08/15/2016. 
By default: "shittest is the main function under windows; "dumbtest" is the 
main function under Linux.

"""

import glob 
import os 
from Pipeline import pipeline_zstacks


def main():
    
    """
    OK this part already works. 
    """ 
    hroot = 'X:\Zebrafish_ispim/'
    abspath = os.path.abspath(hroot)
    aq_date = '/2016-05-18\\'
    fd = abspath + aq_date
    fd_list = glob.glob(fd+'*') # list all the tiff files in the folder  
    work_folder = fd_list[0] + '\\'
    print(work_folder)
    
    tpflags = 'TP'
    pz = pipeline_zstacks(work_folder, tpflags)
#     pz.zstack_prepro(0)
    pz.zstack_tseries()

    
    
    
   

if __name__ == '__main__':
    main()

