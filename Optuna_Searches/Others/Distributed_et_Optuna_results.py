import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate, GroupKFold, cross_val_score
from Spectra_Preparation_Functions import *
import optuna

df = pd.read_csv("../../data/current_clean_spectrum.csv")

wavelength_df = prepare_wavelength_df(df, 'Absorbance')

def objective(trial):

    # Prepare data for ML
    wavelength_copy = wavelength_df.copy()
    X = wavelength_copy.drop(['Status', 'SurID'], axis=1)
    y = wavelength_copy['Status']
    groups = wavelength_copy['SurID']

    # Classifier and cross-validation setup
    cv = GroupKFold(n_splits=10)

    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    max_depth = trial.suggest_int('max_depth', 2, 100, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 1.0)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.01, 0.5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    max_samples = trial.suggest_float('max_samples', 0.01, 1.0) if bootstrap else None

    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
        random_state=1234
    )
    # Perform cross-validation
    #scores = cross_validate(et, X, y, groups=groups, cv=cv, scoring='accuracy')
    scores = cross_val_score(et, X, y, groups=groups, cv=cv, scoring='accuracy')

    # Return the average accuracy across all folds
    return np.mean(scores)

results_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
results_df.to_csv("../../data/et_optuna.csv")

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="distributed-et", storage="mysql://root@localhost/example"
    )
    
    results_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    results_df.to_csv("../../data/et_optuna.csv")
