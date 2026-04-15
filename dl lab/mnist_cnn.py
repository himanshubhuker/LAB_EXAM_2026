import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1,28,28,1) / 255.0
x_test  = x_test.reshape(-1,28,28,1) / 255.0

# Model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5)

# Test
model.evaluate(x_test, y_test)





## Cifer 10 cnn

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)



## matrix

from sklearn.metrics import confusion_matrix
import numpy as np

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test.flatten(), y_pred)
print(cm)


# BEGIN

# LOAD dataset (MNIST / CIFAR10)

# NORMALIZE images

# RESHAPE images (if needed)

# CREATE CNN model:
#     ADD Conv2D layer
#     ADD MaxPooling layer
#     ADD Flatten layer
#     ADD Dense layer
#     ADD Output layer

# COMPILE model using adam optimizer

# TRAIN model

# EVALUATE model

# PREDICT test data

# CONVERT predictions using argmax

# GENERATE confusion matrix

# END