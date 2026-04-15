import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test / 255.0

x_train = x_train[..., None]
x_test  = x_test[..., None]

# Train only normal data (exclude 9)
x_train = x_train[y_train != 9]

# Autoencoder
model = keras.Sequential([
    layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(28,28,1)),
    layers.MaxPooling2D(2),

    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2),

    layers.Conv2DTranspose(16, 2, strides=2, activation='relu'),
    layers.Conv2DTranspose(1, 2, strides=2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(x_train, x_train, epochs=5)

# Reconstruction
recon = model.predict(x_test)

# Error
error = np.mean((x_test - recon)**2, axis=(1,2,3))

# Threshold
threshold = error.mean() + 2*error.std()

# Detect anomaly (9)
y_pred = (error > threshold).astype(int)
y_true = (y_test == 9).astype(int)

# Accuracy
print("Accuracy:", np.mean(y_pred == y_true))



# BEGIN

# LOAD dataset

# NORMALIZE images

# REMOVE anomaly class

# CREATE autoencoder:
#     Encoder
#     Decoder

# TRAIN model

# RECONSTRUCT test data

# CALCULATE reconstruction error

# SET threshold

# IF error > threshold:
#     anomaly
# ELSE:
#     normal

# EVALUATE accuracy

# END