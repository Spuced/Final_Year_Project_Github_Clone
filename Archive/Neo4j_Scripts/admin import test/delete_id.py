#!/usr/bin/env python3

import pandas as pd

# Read the specta
exosomes = pd.read_csv('../../data/exosome_edges.csv',index_col=0)

exosomes.to_csv('test.csv', index=False)