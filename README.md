Image_toolbox
==========================================================

Source code instructions 

-----------------------------------------------------------

Slave branch is for windows.

    This is Dan's imaging toolbox package written in Python. The src/ folder contains:
    dumbtest.py # This is the most frequently updated file. Works like a main() function for testing other not well-developed files.   
    Cell_extract.py # This one utilizes skimage package to extract blobs (nuclei) from stacks of fluorescent images. 
    Preprocessing.py # This file contains all the preprocessing procedures for the raw image, which might be blurred and/or have drifting problems. 
    common_funcs.py # this file contains small functions that are shared among classes. Functions here are not organized as a class.
