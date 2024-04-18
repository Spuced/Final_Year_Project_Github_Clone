#!/usr/bin/env python3

# https://pubs.acs.org/doi/10.1021/acsomega.3c05780
# This uses the model architecture from the above study

import pandas as pd

#spectra_df = pd.read_csv("../data/exosomes.raw_spectrum_400-1800.csv")
spectra_df = pd.read_csv("../data/scaled_cleaned_cnn_spectra.csv")  #lam10**5 p 0.001 w=21 poly=2

#### **Train a Neural Network on the full spectrum**

wavelength_df = spectra_df.pivot(index='SpecID', columns='WaveNumber', values='Absorbance').reset_index()
wavelength_df.columns.name = None
surface_and_statuses = spectra_df[['SpecID', 'Status', 'SurID']].drop_duplicates()
wavelength_df = pd.merge(wavelength_df, surface_and_statuses, on='SpecID')
wavelength_df = wavelength_df.set_index('SpecID')
wavelength_df.head()
X = wavelength_df.drop(columns=['Status'])
y = pd.get_dummies(wavelength_df['Status'])  # One-hot encode target variable

#### **CNN Training**
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import GroupShuffleSplit

# Split data into training and testing sets ensuring no overlap in SurID
group_kfold = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=1234)

# Outline the groups for GroupKFold
groups = X['SurID']
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train_temp, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_temp, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Re-apply GroupShuffleSplit on the preliminary training set to further split it into training and validation
group_kfold_val = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=1234)  # Adjust the test_size as necessary
for train_idx, val_idx in group_kfold_val.split(X_train_temp, y_train_temp, X_train_temp['SurID']):
    X_train, X_val = X_train_temp.iloc[train_idx], X_train_temp.iloc[val_idx]
    y_train, y_val = y_train_temp.iloc[train_idx], y_train_temp.iloc[val_idx]

# Remove the SurID column as it should not be used as a feature for training
X_train = X_train.drop(columns=['SurID'])
X_test = X_test.drop(columns=['SurID'])
X_val = X_val.drop(columns=['SurID'])

# Reshape data for 1D convolution
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, UpSampling1D, Flatten, Dense, Add
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    # Initial input layer
    inputs = Input(shape=input_shape)

    # First Conv1D block
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)

    # Shortcut connection for the first block
    shortcut = Conv1D(filters=72, kernel_size=1, strides=2, padding='same')(x)

    # First block repeated twice
    for _ in range(2):
        x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)

        # Adding the shortcut
        x = Add()([x, shortcut])
        # Preparing shortcut for the next block
        shortcut = Conv1D(filters=48, kernel_size=1, strides=2, padding='same')(x)

    # Implementing upsampling followed by Conv1D to simulate ConvTranspose1D
    x = UpSampling1D(size=2)(x)  # Upsample the sequence
    x = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu')(x)  # Apply convolution

    # Shortcut connection for the ConvTranspose1D block
    x = Add()([x, shortcut])

    # Final Conv1D blocks
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)

    # Conv1D to 1x512
    x = Conv1D(filters=1, kernel_size=1, padding='same', activation='relu')(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    # Construct and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

input_shape = (X_train.shape[1], 1)
model = build_model(input_shape)
model.summary()
