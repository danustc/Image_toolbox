#!/home/sillycat/Softwares/anaconda3/bin/python
"""
Created by Dan Xie on 10/17/2016. The first one edited with Atom!
"""

from pyexcel_xls import save_data, get_data
import json
import numpy as np

def read_xls_column(xls_file, col_tag):
    """
    read one or more columns from the xls file.
    """
    data = get_data(xls_file)
    
