import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.optimizers import Adam

# Based on https://pubs.acs.org/doi/full/10.1021/acsnano.9b09119

spectra_df = pd.read_csv("../data/scaled_cleaned_cnn_spectra.csv")  #lam10**5 p 0.001 w=21 poly=2
spectra_df

#### **Prepare the input features**
wavelength_df = spectra_df.pivot(index='SpecID', columns='WaveNumber', values='Absorbance').reset_index()
wavelength_df.columns.name = None
surface_and_statuses = spectra_df[['SpecID', 'Status', 'SurID']].drop_duplicates()
wavelength_df = pd.merge(wavelength_df, surface_and_statuses, on='SpecID')
wavelength_df = wavelength_df.set_index('SpecID')
wavelength_df.head()
X = wavelength_df.drop(columns=['Status'])
y = pd.get_dummies(wavelength_df['Status'])  # One-hot encode target variable

# Outline the groups for GroupKFold
groups = X['SurID']
X = X.drop(columns='SurID')

n_splits=5
group_kfold = GroupKFold(n_splits=n_splits)

# Model Configuration
def create_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation loop
fold_no = 1
for train_index, test_index in group_kfold.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = create_model(input_shape=(X_train.shape[1], 1))
    print(f'Training fold {fold_no}/{n_splits}...')
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    fold_no += 1


# model = Sequential([
#     Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     MaxPooling1D(pool_size=2),

#     Conv1D(filters=64, kernel_size=3, activation='relu'),
#     MaxPooling1D(pool_size=2),

#     Conv1D(filters=128, kernel_size=3, activation='relu'),
#     MaxPooling1D(pool_size=2),

#     Conv1D(filters=256, kernel_size=3, activation='relu'),
#     MaxPooling1D(pool_size=2),

#     Conv1D(filters=512, kernel_size=3, activation='relu'),
#     MaxPooling1D(pool_size=2),

#     Flatten(),
#     Dense(4096, activation='relu'),
#     Dense(10, activation='relu'),
#     Dropout(0.4),

#     Dense(3, activation='softmax')
# ])

# # Custom optimizer with specified learning rate and decay, based on the parameters in the paper
# learning_rate = 9.8e-4
# decay_rate = 0.2
# optimizer = Adam(learning_rate=learning_rate, decay=decay_rate / 200)

# # Compile model with custom optimizer
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=200, batch_size=500, validation_data=(X_val, y_val))