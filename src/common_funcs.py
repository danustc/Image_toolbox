'''
Created by Dan on 08/02/16
This file contains several small functions shared among all the classes.
'''

import numpy as np

def gaussian2D(am, cx, cy, dx, dy, ofst):
    # am: amplitude
    # cx, cy: center x, y
    # dx, dy: width in x, y direction 
    # ofst: the offset constant 
    return lambda x,y: am*np.exp(-((x-cx)/dx)**2 -((y-cy)/dy)**2)+ofst


    



    