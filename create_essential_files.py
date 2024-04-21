''' 
This is a short script that will create some common permutations of the raw spectral data, that are used in other scripts.

The raw file "exosomes.raw_spectrum_1.csv" should be located in the data directory in the root of the repo
'''

# Import the libraries
from Cleaning_and_Evaluation import *
import pandas as pd

# Limit the spectra to the effective 400-1800 range
print("Limiting the spectra to the essential range 400-1800 cm-1")
df = pd.read_csv("./data/exosomes.raw_spectrum_1.csv")

effective_df = df[(df['WaveNumber'] >= 400) & (df['WaveNumber'] <= 1800)]

effective_df.to_csv("./data/exosomes.raw_spectrum_400-1800.csv", index=False)

# Clean the spectra using the default parameters in case some files require pre-cleaned spectra.
print("Cleaning the spectra with default parameters")
spectra_cleaning(df)
df.to_csv("./data/current_clean_spectrum.csv", index=False)