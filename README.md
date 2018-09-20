# Image\_toolbox
Last update: 09/20/2018 by [@danustc](https://github.com/danustc/) 
## Source code instructions

### -------------------Package dependence------------------
**tifffile**: developed by [Christoph Gohlke](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html)         
**scikit-image**:  version 0.14 or later    
**scikit-learn**: currently 0.19.2, but other versions may work too

### -------------------Main pipelines---------------------
**Pipeline.py** : The main pipeline of data processing. Each run pre-processes the data from one fish. The t-stacks 
should be pre-aligned with the _MultiStackReg_ plugin in FIJI. Cell extraction is performed on selected slices, merged and then propagated through the whole stack to calculate the raw F values. If you need a deblur prior to cell extraction, please set _sig > 0_ (for instance, 4), so a Gaussian-shaped artificial PSF will be generated to deconvolve the raw images.    
**Pipeline\_zstacks.py**: The pipeline that processes z-stacks instead of t-stacks. Cells are extracted in each slice and saved as a dictionary.


### ------------------Modules (src folder) ------------------------

####  **preprocessing**
* **crossalign\_pipeline.py**: cross align T-stacks to Z-stacks and merge segmented cell informations from different stacks into one file. Detection redundancy would be removed in this pipeline.
* **stack\_operations.py**: some basic tiff stack operations: cropping, thresholding, binning, etc.
* **segmentation.py**: segmentating regions of interest (ROI) using blob detection in scikit-image.
* **drift\_correction.py**: correlation-based drift-correction, still under test.
####  **visualization**
* **signal\_plot.py**: plot \Delta F/F signals in different styles.
* **stat\_present.py**: visualization of statistical analysis results, including PCA, ICA, K-means and regressions.
* **brain\_navigation.py**: display raw image slices or volumes, highlight marked neurons.
* **cluster\_navigation.py**: A 2D visualization of clustering results of 3D data.
####  **analysis**
* **df\_f.py**: Core functions of \Delta F/F calculation, based on the 2011 Nature Protocol paper. Exponential filtering and Bayesian inference peak detection are algo included in this module.
* **dff\_pipeline.py**: calculate \Delta F/F over calcium signals of a population of neurons, do edge artifact correction and smoothing if necessary.
* **network\_ui.py**: an interactive UI for individual dataset analysis and visualization.
* **Analysis.py**: The core class of data analysis, which does activity soring, shuffling and background suppression. The class can be loaded by other analysis classes, such as **pca\_analysis**
* **spectral.py**: The analysis on the frequency domain.

####  **registration**
* **anatomy\_annotation.py**: annotate neuronal identities of selected neurons based on image registration outputs. Image stacks should be registered to the Z-brain template.
* **maskdb\_parsing.py**: anatomical annotation of cells based on 294 masks in the [Z-brain atlas](https://engertlab.fas.harvard.edu/Z-Brain/#/home/).  

#### **shared\_funcs**
Some numerical, string and image processing functions shared by other modules.
