#!/usr/bin/env python3

import pandas as pd

# Read the specta
exosomes = pd.read_csv('../../data/exosome_edges.csv')
exosomes[':TYPE'] = "Distance"

exosomes.to_csv('test.csv', index=False)