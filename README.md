# CA4021 Data Science Final Year Project

**Edward Bolger** (20364133) Email: [edward.bolger25@mail.dcu.ie](edward.bolger25@mail.dcu.ie)

**Stan Goodwin** (20449042) Email: [stan.goodwin6@mail.dcu.ie](stan.goodwin6@mail.dcu.ie)

## Abstract

Current studies into the use of machine learning for the classification of disease, based on the Raman spectra of exosomes have claimed near perfect classification accuracy. In this project we explain a simple validation error that can result in dramatically overestimated model performance. We then outline some novel spectral processing techniques utilising graph databases and feature selection, that improve the actual generalisation ability of these models.

### Repository Structure:

- [**`archive`**](/Archive/): Contains old notebooks used in development that have been superceded.
- [**`Baseline_Approach`**](/Baseline_Approach/): Contains cleaning experiments for the full wavelength and for peak features using KFold cross-validation.
- [**`Cleaning_and_Evaluation`**](/Cleaning_and_Evaluation/): A **Python package** containing cleaning and evaluation functions used in all notebooks.
- [**`Feature_Selection`**](/Feature_Selection/): Contains notebooks for exploring a variety of feature selection methods.
- [**`Final_Results`**](/Final_Results/): Contains notebooks that evaluate the best approaches, outlied in the report, found through GridSearch and Bayesian Search.
- [**`Graph_Engineering`**](/Graph_Engineering/): Contains the notebooks used to create the graph structures, and extract metrics from Neo4j.
- [**`GroupKFold_Baseline_Approach`**](/GroupKFold_Baseline_Approach/): Contains cleaning experiments for the full wavelength and for peak features using GroupKFold cross-validation.
- [**`Neural Networks`**](/Neural%20Networks/): Contains experiments to classify spectra using neural approaches.
- [**`Notes`**](/Notes/): A folder used for shorthand notes and observations.
- [**`Outlier_Detection`**](/Outlier_Detection/): Contain different methods for outlier detection, including [PageRank Filter](/Outlier_Detection/PageRank_Filter_Before_Clean.ipynb).
- [**`Parameter_Searches`**](/Parameter_Searches/): Contains experiments for a variety of feature selection approaches.
- [**`data`**](/data/): Place raw data into this folder and then run [create_essential_files.py](/create_essential_files.py).
- [**`docs`**](/docs/): Contains project documentation.



Run [create_essential_files.py](/create_essential_files.py) after placing raw data into the [data](/data/) folder to create files need to run notebooks in this project. Package requirements are detailed in [requirements](/requirements.md).

Check the respective folders for more details.

#### Effect of PageRank on Spectra Within a Surface

[Most Central Spectra](/images/)
