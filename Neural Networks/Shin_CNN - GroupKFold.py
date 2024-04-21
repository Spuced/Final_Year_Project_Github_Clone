import pandas as pd

# Based on https://pubs.acs.org/doi/full/10.1021/acsnano.9b09119

spectra_df = pd.read_csv("../data/scaled_cleaned_cnn_spectra.csv")  #lam10**5 p 0.001 w=21 poly=2
spectra_df

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
from tensorflow.keras.optimizers import Adam

# Split data into training and testing sets ensuring no overlap in SurID
group_kfold = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=1234)

# Outline the groups for GroupKFold
groups = X['SurID']
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train_temp, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_temp, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Re-apply GroupShuffleSplit on the preliminary training set to further split it into training and validation
group_kfold_val = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=1234)
for train_idx, val_idx in group_kfold_val.split(X_train_temp, y_train_temp, X_train_temp['SurID']):
    X_train, X_val = X_train_temp.iloc[train_idx], X_train_temp.iloc[val_idx]
    y_train, y_val = y_train_temp.iloc[train_idx], y_train_temp.iloc[val_idx]

# Remove the SurID
X_train = X_train.drop(columns=['SurID'])
X_test = X_test.drop(columns=['SurID'])
X_val = X_val.drop(columns=['SurID'])

# Reshape data for 1D convolution
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=512, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(4096, activation='relu'),
    Dense(10, activation='relu'),
    Dropout(0.4),

    Dense(3, activation='softmax')
])

# Custom optimizer with specified learning rate and decay, based on the parameters in the paper
learning_rate = 9.8e-4
decay_rate = 0.2
optimizer = Adam(learning_rate=learning_rate, decay=decay_rate / 200)

# Compile model with custom optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=500, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
