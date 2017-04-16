"""
Created by Dan Xie on 08/15/2016. 
By default: "shittest is the main function under windows; "dumbtest" is the 
main function under Linux.

"""

import glob 
import os 
from Pipeline import pipeline_tstacks


def main():
    
    """
    OK this part already works. 
    """ 
    hroot = 'D:\Data/'
    abspath = os.path.abspath(hroot)
    aq_date = '/2016-09-28\\'
    fd = abspath + aq_date
    fd_list = glob.glob(fd+'*TS*') # list all the tiff files in the folder
    tpflags = 'ZP'
    fd_list.sort(key=os.path.getmtime)  
    for work_folder in fd_list:
        work_folder = work_folder+ '\\'
        print(work_folder)
     
        pz = pipeline_tstacks(work_folder, tpflags)
        pz.tstack_zseries(deblur = 30, align = True, ext_all = False)
        
def shit1():
    """
    Correct a stack of shift
    How to add argc, argv like what we do in c/c++ ?
    """
    hroot = 'D:\Data/'
    abspath = os.path.abspath(hroot)
    aq_date = '/2016-09-29\\'
    fd = abspath + aq_date  
    fd_list = glob.glob(fd+'*TS*')
    for work_folder in fd_list:
        work_folder += '\\'
        pt = pipeline_tstacks(work_folder, zp_flags='ZP')
        pt.tstack_zseries(deblur = 30, align = True, ext_all = False)
    
    
    
    

if __name__ == '__main__':
    shit1()

