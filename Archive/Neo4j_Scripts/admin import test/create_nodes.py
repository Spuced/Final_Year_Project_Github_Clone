#!/usr/bin/env python3

import pandas as pd

# Read the specta
exosomes = pd.read_csv('../../data/exosomes.raw_spectrum_1.csv')

# Extract status of each sample
samples = exosomes.groupby('SpecID')['Status'].first()

# Write to csv
samples.to_csv('../../data/exosome_nodes.csv')