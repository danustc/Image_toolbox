"""
Created by Dan Xie on 08/15/2016. 
By default: "shittest is the main function under windows; "dumbtest" is the 
main function under Linux.

"""

import glob 
import os 



def main():
    
    """
    OK this part already works. 
    """ 
    hroot = 'X:\Zebrafish_ispim/'
    abspath = os.path.abspath(hroot)
    print(abspath)
    aq_date = '/2016-05-18\\'
    fd = abspath + aq_date
    
    print(fd)
        
    fd_list = glob.glob(fd+'*') # list all the tiff files in the folder  
    
    for fd in fd_list:
        print(fd) 
    
    
   

if __name__ == '__main__':
    main()

