"""
Created by Dan on 08/22/2016
Contains all the string functions (mostly file name parsing)
"""

import ntpath


def path_leaf(path):
    """
    A tiny function for splitting filename from a long path, always the last layer, folder or not
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
    # done with path_leaf.


def number_strip(name_str, delim = '_', ext = True):
    """
    Strip the number from the filename string.
    Untested.
    """
    if(ext):
        khead = name_str.split('.')[0]
    else:
        khead = name_str
    klist = khead.split(delim) 
    knum = [km for km in klist if km.isdigit()]
    korder =knum[-1] # usually the last one represent 
    # how tor
    
    return korder
    # done with number_strip.
    

def fname_postfix(dph, postfix):
    """
    insert a postfix into a filename before the numbering digits.
    """
    head, tail = ntpath(dph)
    ftrunk, fext = tail.split('.') # split the name body and extension
    
    if(ftrunk[-1].isdigit()):
        ii = -2
        while(ftrunk[ii].isdigit()):
            ii-=1
        
        ft_h = ftrunk[:(ii+1)]
        ft_t = ftrunk[(ii+1):]
        
        pfx_name = ''.join([head, ft_h, postfix, ft_t])
        
    else:
        pfx_name = ''.join([head, ftrunk, postfix])
    

    return '.'.join(pfx_name, fext)
    # done with fname_postfix


