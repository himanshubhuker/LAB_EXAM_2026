import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv("data.csv")
series = data["meantemp"].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()
series = scaler.fit_transform(series)

# Create sequences
def create_data(series, window=30):
    X, y = [], []
    for i in range(len(series)-window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

X, y = create_data(series)

# Split
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = keras.Sequential([
    layers.LSTM(64, input_shape=(30,1)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=10)

# Test
model.evaluate(X_test, y_test)

# BEGIN

# LOAD dataset

# SELECT temperature column

# NORMALIZE data

# CREATE sequences using window

# SPLIT into train and test

# CREATE LSTM model

# COMPILE model

# TRAIN model

# PREDICT values

# EVALUATE model

# END