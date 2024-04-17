# Set up the dependencies
import pandas as pd
import numpy as np
from pybaselines.whittaker import asls, iasls, airpls, psalsa
from pybaselines.polynomial import poly
from sklearn.model_selection import KFold, GroupKFold, cross_validate 
from scipy.signal import savgol_filter

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
def snv_normalise(absorbances):
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

def spectra_cleaning(df, **kwargs):

    # Retrieve keyword arguments with default values if they are not provided
    despike = kwargs.get('despike', True)
    baseline_correct = kwargs.get('baseline_correct', True)
    smoothing = kwargs.get('smoothing', True)
    scaling = kwargs.get('scaling', 'vector')
    despike_ma = kwargs.get('despike_ma', 10)
    despike_threshold = kwargs.get('despike_threshold', 7)
    lam = kwargs.get('lam', 10**7)
    p = kwargs.get('p', 0.01)
    window_size = kwargs.get('window_size', 21)
    poly_order = kwargs.get('poly_order', 3)

    # Whitaker Hayes Despiking
    if despike:
        df['Absorbance'] = df.groupby('SpecID')['Absorbance'].transform(lambda x: despike_group(x, ma=despike_ma, threshold = despike_threshold))

    # Asymmetric Least Squares Baseline Correction
    if baseline_correct:
        df['Absorbance'] = df.groupby('SpecID')['Absorbance'].transform(lambda x: x - asls_baseline_correction(x, lam=lam, p=p))

    # Savitsky-Golay Smoothing
    if smoothing:
        df['Absorbance'] = df.groupby('SpecID')['Absorbance'].transform(lambda x: savgol_filter(x, window_size, poly_order, deriv=0))

    # Normalisation
    if scaling:
        if scaling == 'normal':
            df['Absorbance'] = df.groupby('SpecID')['Absorbance'].transform(lambda x: normalise(x))
        elif scaling == 'vector':
            df['Absorbance'] = df.groupby('SpecID')['Absorbance'].transform(lambda x: vector_normalise(x))
        else:
            df['Absorbance'] = df.groupby('SpecID')['Absorbance'].transform(lambda x: svn_normalise(x))

# This function pivots the data, converting each WaveNumber to a column
def prepare_wavelength_df(df, absorbance_col='Absorbance'):

    # Pivot the DataFrame to get wavelengths as columns and absorbance values
    wavelength_df = df.pivot(index='SpecID', columns='WaveNumber', values=absorbance_col).reset_index()
    wavelength_df.columns.name = None

    # Merge with the statuses and surfaces based on SpecID
    statuses_and_surface = df[['SpecID', 'SurID', 'Status']].drop_duplicates()
    wavelength_df = pd.merge(wavelength_df, statuses_and_surface, on='SpecID')

    # Set SpecID as the index
    wavelength_df = wavelength_df.set_index('SpecID')

    return wavelength_df

# Evaluate a model with either KFold or GroupKFold
def evaluate_model(df, model, groupkfold=True, n_jobs=-1):

    # Set the Surfaces as groups
    groups = df['SurID']
    X = df.drop(['Status', 'SurID'], axis=1)
    y = df['Status']

    # Perform GroupKFold Cross-Validation
    if groupkfold:

        # Using GroupKFold for classification tasks
        cv = GroupKFold(n_splits=10)

    # Perform KFold Cross-Validation
    else:
        cv = KFold(n_splits=10, random_state=1234, shuffle=True)

    # Cross Validate
    scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], n_jobs=-1)

    # Displaying the results
    print(f"{model.__class__.__name__} Cross-Validation Accuracy: {np.mean(scores['test_accuracy']):.4f} +/- {np.std(scores['test_accuracy']):.4f}")
    print(f"{model.__class__.__name__} Cross-Validation Precision: {np.mean(scores['test_precision_macro']):.4f} +/- {np.std(scores['test_precision_macro']):.4f}")
    print(f"{model.__class__.__name__} Cross-Validation Recall: {np.mean(scores['test_recall_macro']):.4f} +/- {np.std(scores['test_recall_macro']):.4f}")
    print(f"{model.__class__.__name__} Cross-Validation F1-Score: {np.mean(scores['test_f1_macro']):.4f} +/- {np.std(scores['test_f1_macro']):.4f}")
