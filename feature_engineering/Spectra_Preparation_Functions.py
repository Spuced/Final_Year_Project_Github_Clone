
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
    ysb = np.diff(ys)  # Differentiated intensity values
    median_y = np.median(ysb)  # Median of the intensity values
    mad_y = np.median(np.abs(ysb - median_y))  # Median absolute deviation
    modified_z_scores = 0.6745 * (ysb - median_y) / mad_y
    return np.concatenate([[0], modified_z_scores])  # Include a placeholder for the first element

def fixer(y, ma, threshold):
    spikes = np.abs(modified_z_score(y)) > threshold
    y_out = np.copy(y)
    
    for i in np.where(spikes)[0]:  # Loop over indices where spikes occur
        w_start = max(i - ma, 0)
        w_end = min(i + ma + 1, len(y))
        valid_indices = ~spikes[w_start:w_end]  # Indices within the window that are not spikes

        if np.any(valid_indices):
            valid_y = y[w_start:w_end][valid_indices]
            y_out[i] = np.mean(valid_y)
    
    return y_out

def despike_group(absorbances, ma=20, threshold=7):
    despiked_absorbance = fixer(absorbances.to_numpy(), ma=ma, threshold=threshold)
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
