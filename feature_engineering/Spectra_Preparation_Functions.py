
# Set up the dependencies
import pandas as pd
import numpy as np
from pybaselines.whittaker import asls, iasls, airpls, psalsa
from pybaselines.polynomial import poly

# This function pivots the data, converting each WaveNumber to a column
def prepare_wavelength_df(df, absorbance_col, status_col='Status'):

    # Pivot the DataFrame to get wavelengths as columns and absorbance values
    wavelength_df = df.pivot(index='SpecID', columns='WaveNumber', values=absorbance_col).reset_index()
    wavelength_df.columns.name = None

    # Merge with the statuses based on SpecID
    # Include the SurID to perform GroupKFold CV
    statuses_and_surface = df[['SpecID', 'SurID', status_col]].drop_duplicates()
    wavelength_df = pd.merge(wavelength_df, statuses_and_surface, on='SpecID')

    # Set SpecID as the index
    wavelength_df = wavelength_df.set_index('SpecID')

    return wavelength_df

# Scale each spectra to its highest peak 
def normalise(absorbances):
    max_value = np.max(absorbances)
    normalised_absorbances = absorbances / max_value
    return normalised_absorbances

# Scale each spectra to their euclidean norm
def vector_normalise(absorbances):
    l2_norm = np.sqrt(np.sum(absorbances**2))  # Calculate the euclidean norm
    normalized_absorbances = absorbances / l2_norm
    return normalized_absorbances

# Scale each spectra to the mean and standard deviation of the absorbance
def svn_normalise(absorbances):
    mean = absorbances.mean()
    std = absorbances.std()
    return (absorbances - mean) / std

# Perform min-max scaling on the spectra
def min_max_scale(absorbances):
    min_value = np.min(absorbances)
    max_value = np.max(absorbances)
    return (absorbances - min_value) / (max_value - min_value)

def modified_z_score(ys):
    ysb = np.diff(ys) # Differentiated intensity values
    median_y = np.median(ysb) # Median of the intensity values
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ysb]) # median_absolute_deviation of the differentiated intensity values
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ysb] # median_absolute_deviationmodified z scores
    return modified_z_scores
    
# The next function calculates the average values around the point to be replaced.
def fixer(y, ma, threshold):
    spikes = abs(np.array(modified_z_score(y))) > threshold
    y_out = y.copy()
    for i in np.arange(len(spikes)):
        
        if spikes[i] != 0:
            # Calculate the window range, ensuring it stays within the bounds of the spectrum
            w_start = max(i - ma, 0)
            w_end = min(i + ma + 1, len(y))
            w = np.arange(w_start, w_end)
            
            valid_w = w[w < len(spikes)]  # Ensure w doesn't go beyond the length of spikes
            
            # Indices within the window that do not correspond to spikes
            valid_indices = valid_w[~spikes[valid_w]]
            
            # If there are valid indices, calculate the mean of 'y' over these indices
            if len(valid_indices) > 0:
                y_out[i] = np.mean(y[valid_indices])
            else:
                y_out[i] = y[i]
    return y_out

def despike_group(absorbances, ma=20, threshold=7):
    absorbance_data = absorbances.to_numpy()
    despiked_absorbance = fixer(absorbance_data, ma=ma, threshold=threshold)
    return despiked_absorbance

def asls_baseline_correction(x, lam, p):
        corrected, _ = asls(x, lam=lam, p=p)
        return corrected

def iasls_baseline_correction(x, lam, p, lam_1):
        corrected, _ = iasls(x, lam=lam, p=p, lam_1=lam_1)
        return corrected

def airpls_baseline_correction(x, lam):
        corrected, _ = airpls(x, lam=lam)
        return corrected

def poly_baseline_correction(x, poly_order):
        corrected, _ = poly(x, poly_order=poly_order)
        return corrected

def psalsa_baseline_correction(x, lam, p):
        corrected, _ = psalsa(x, lam=lam, p=p)
        return corrected
