# Image_toolbox

## Source code instructions

### -------------------Package dependence------------------
tifffile: developed by [Christoph Gohlke](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html)         
scikit-image (version 0.14 or later)

### -------------------Main pipelines---------------------

### ------------------Modules (src folder) ------------------------

####  **preprocessing**
* **crossalign\_pipeline.py**: cross align T-stacks to Z-stacks.
* **stack\_operations.py**: some basic tiff stack operations: cropping, thresholding, binning, etc.
* **segmentation.py**: segmentating regions of interest (ROI) using blob detection in scikit-image.
####  **visualization**
* **signal\_plot.py**: plot \Delta F/F signals in different styles.
* **stat\_present.py**: visualization of statistical analysis results, including PCA, ICA, K-means and regressions.
* **brain\_navigation.py**: display raw image slices or volumes, highlight marked neurons.
####  **networks**
* **pca\_sorting.py**: PCA-based neuronal activity sorting.
* **dff\_pipeline.py**: calculate \Delta F/F over calcium signals of a population of neurons, do edge artifact correction and smoothing if necessary.
* **network\_ui.py**: an interactive UI for individual dataset analysis and visualization.
####  **registration**
* **anatomy\_annotation.py**: annotate neuronal identities of selected neurons based on image registration outputs. Image stacks should be registered to the Z-brain template.
