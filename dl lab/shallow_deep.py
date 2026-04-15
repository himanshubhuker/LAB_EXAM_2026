import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
(xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

# Preprocess
xtrain = xtrain.reshape(-1, 784) / 255.0
xtest = xtest.reshape(-1, 784) / 255.0

# Model function
def build_model(deep=False):
    model = keras.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
    
    if deep:
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
    
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Shallow model
model1 = build_model(deep=False)
model1.fit(xtrain, ytrain, epochs=5)
model1.evaluate(xtest, ytest)

# Deep model
model2 = build_model(deep=True)
model2.fit(xtrain, ytrain, epochs=5)
model2.evaluate(xtest, ytest)

from sklearn.metrics import confusion_matrix

y_pred = model1.predict(xtest)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(ytest, y_pred)
print(cm)


# BEGIN

# LOAD MNIST dataset

# NORMALIZE data (divide by 255)

# RESHAPE images to 784 features

# FUNCTION build_model(deep):
#     CREATE model
#     ADD Dense layer (512, relu)

#     IF deep == TRUE:
#         ADD Dense (256, relu)
#         ADD Dense (128, relu)

#     ADD output layer (10, softmax)

#     COMPILE model using adam optimizer
#     RETURN model

# CREATE shallow_model = build_model(FALSE)

# TRAIN shallow_model

# EVALUATE shallow_model

# CREATE deep_model = build_model(TRUE)

# TRAIN deep_model

# EVALUATE deep_model

# PREDICT test data

# CONVERT predictions using argmax

# GENERATE confusion matrix

# END