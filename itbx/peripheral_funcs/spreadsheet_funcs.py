#!/home/sillycat/Softwares/anaconda3/bin/python
"""
Created by Dan Xie on 10/17/2016. The first one edited with Atom!
Last update: 10/26/2016, using pandas instead of read xls 
"""

import pandas as pd 
import numpy as np


def read_xls_column(xls_file, col_tag, sheet_name = None):
    """
    read one or more columns from the xls file.
    """
    xl = pd.ExcelFile(xls_file) 
    sns = xl.sheet_names # extract the sheet names 
    if sheet_name is None:
        df = xl.parse(sns[0])
    else:
        df = xl.parse(sheet_name)
        # have a dataframe 
    
    data = df[col_tag].values()
    return data


def write_xls_column(data_mat, xls_file, col_tags, sheet_name = 'Sheet1'): 
    """
    write one or more columns into the xls file.
    and this is my first time using try-except structure! :) 
    """
    writer = pd.ExcelWriter(xls_file, engine = 'xlsxwriter')

    try:
        df = pd.DataFrame(data = data_mat,columns = col_tags)
        df.to_excel(writer, sheet_name)
        writer.save()
    except ValueError:
        print("columns of the matrix and number of the column tags mismatch.")
    