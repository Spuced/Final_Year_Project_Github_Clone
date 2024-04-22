# Graph Engineering

Here you will find a collection of Jupyter notebooks showcasing different methods of constructing graph databases for feature extraction to be used in the machine classification of exosomes based on their Raman Spectra.

## Graph Databases

### Peak Grids Graph
In this notebook, we construct a graph using peaks extracted from spectra. Each peak serves as a node. Relationships between peaks are established based on two conditions: every peak within a sample is connected, and peaks within the same grid square form connections. Relationships are weighted by distance, ensuring closer peaks have stronger connections.

### Biomarker Peak Ranges Graph
This notebook presents a graph with two node types: Spectra and PeakRange. Spectra nodes represent spectra samples, while PeakRange nodes denote ranges of WaveNumber values with chemical or biological significance. Relationships exist between Spectra and PeakRange nodes if a peak falls within the defined range, with relationship weights based on the peak's absorbance.

### Gaussian Kernel Graph
Here, each spectra serves as a node, forming a fully connected graph. Relationship weights between nodes are determined by the Euclidean distance between spectra, transformed using a Gaussian kernel function to measure similarity.

## Evaluation Methods

Each graph is evaluated using two techniques:

1. **KFold Cross Validation**: Data is divided into K folds, with each fold serving as a validation set while the rest are used for training. This process is repeated K times, ensuring all data is used for both training and validation.

2. **GroupKFold Cross Validation**: Similar to KFold, but preserves the grouping of data samples. This is used to prevent data leakage across seperate surfaces.
