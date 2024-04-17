import pandas as pd

#spectra_df = pd.read_csv("../data/exosomes.raw_spectrum_400-1800.csv")
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

# Split data into training and testing sets ensuring no overlap in SurID
group_kfold = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=1234)

# Assuming 'SurID' is a column in X, which is used to group the data
groups = X['SurID']
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train_temp, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_temp, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Re-apply GroupShuffleSplit on the preliminary training set to further split it into training and validation
group_kfold_val = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=1234)  # Adjust the test_size as necessary
for train_idx, val_idx in group_kfold_val.split(X_train_temp, y_train_temp, X_train_temp['SurID']):
    X_train, X_val = X_train_temp.iloc[train_idx], X_train_temp.iloc[val_idx]
    y_train, y_val = y_train_temp.iloc[train_idx], y_train_temp.iloc[val_idx]

# Remove the SurID column if it should not be used as a feature for training
X_train = X_train.drop(columns=['SurID'])
X_test = X_test.drop(columns=['SurID'])
X_val = X_val.drop(columns=['SurID'])

# Reshape data for 1D convolution
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], 1)
# # Define CNN architecture

model = Sequential([
    Conv1D(filters=1028, kernel_size=100, activation='relu', input_shape=(X_train.shape[1], 1)),
    Flatten(),
    Dense(3, activation='softmax')
    # Conv1D(filters=64, kernel_size=3, activation='relu'),
    # MaxPooling1D(pool_size=2),
    # Dropout(0.5),
    # BatchNormalization(),
    
    # Conv1D(filters=128, kernel_size=3, activation='relu'),
    # Conv1D(filters=128, kernel_size=3, activation='relu'),
    # MaxPooling1D(pool_size=2),
    # Dropout(0.5),
    # BatchNormalization(),
    
    # Flatten(),
    # Dense(512, activation='relu'),
    # Dropout(0.5),
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')