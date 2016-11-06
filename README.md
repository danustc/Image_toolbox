# Image_toolbox

## Source code instructions

### -------------------Package dependence------------------
tifffile: developed by [Christoph Gohlke](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html)         
scikit-image (version 0.12 or later)

### -------------------Demos folder ---------------------
Demos.py: contains test codes under Ubuntu.  
Shittest.py: contains test codes under Windows (Because windows is shit!)
### ------------------src folder ------------------------
++ last update: 11/05/2016 by Dan ++

**Alignments.py**: remove the translational drift, based on Fourier transform-cross correlations.   
**Background_correction.py**: Remove low-frequency background in plane.   
**Cell_extract.py**: extrack nuclei using blob detection.   
df_f.py: Calculating \Delta F/F from the raw fluorescence train.  
**Pipeline.py**: The whole set of pre-processing, including background subtraction, drift correction, cell extraction.  The extracted cellular fluorescence and coordinates are saved as .npz files.  
**z_dense.py**: Extract cells from a densely sampled volume, save the cellular coordinates as position references, remove the overcounting across slices, and register the sparsely-sampled z-stacks to the dense z-reference stack.  
**correlations.py**:Calculate pearson correlations between the \Delta F/F profiles between all the pairs of cells inside the imaged stack.
