# CA4021 Data Science Final Year Project

**Edward Bolger** (20364133) Email: [edward.bolger25@mail.dcu.ie](edward.bolger25@mail.dcu.ie)

**Stan Goodwin** (20449042) Email: [stan.goodwin6@mail.dcu.ie](stan.goodwin6@mail.dcu.ie)

## Short Overview

This project aims to test the use of feature engineering and graph databases to classify disease, based on the Raman spectral 'fingerprint' of exosomes, small particles emitted by cells.

### Repository Structure:

- [**`archive`**](./archive): Contains old notebooks that have been superceded.
- [**`docs`**](./docs): Contains project documentation.
- [**`feature_engineering_400-1800cm-1`**](./feature_engineering_400-1800cm-1): Contains the code for spectral cleaning, feature engineering and traditional machine learning, within the 400-1800 cm-1 range.
- [**`feature_engineering`**](./feature_engineering): Contains the code for spectral cleaning, feature engineering and traditional machine learning.
- [**`graph_database`**](./graph_database): Contains the notebooks used to create the graph structures, and extract metrics from Neo4j.
- [**`neo4j_script`**](./neo4j_script): Contains Neo4j queries used to interact with the graph databases.
- [**`notes`**](./notes): A folder used for shorthand notes and observations.

The code in this repo assumes that you have a **`data`** folder in the same directory as the repository where the raw spectral data is stored.

Check the respective folders for more details.
