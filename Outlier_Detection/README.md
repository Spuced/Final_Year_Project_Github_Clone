# Outlier Detection

Here you'll find various methods for detecting and handling outlier or noisy Raman spectra.

## Methods Overview

### Median Spectra
This method involves calculating the median for every spectra in a surface. By identifying the median spectra, outliers or noisy data points can be detected and removed.

### Interquartile Range (IQR)
Using the Interquartile Range (IQR), outlier spectra are identified based on their deviation from the median spectra. Spectra lying outside a defined range are considered outliers and if a certain proporition of a spectra is made up of outliers, it is dropped.

### Gaussian Kernel Graph Outlier Detection
In this approach, a Gaussian Kernel graph is utilized to identify and remove outliers by dropping the least central spectra based on their PageRank score. Two notebooks are provided:
1. **Clean then Build Graph**: Spectra cleaning is performed first, followed by graph construction.
2. **Build Graph then Clean**: Graph construction is performed first, then spectra cleaning.

### Demonstration
Additionally, a notebook is included for visualizing all spectra in a surface.


