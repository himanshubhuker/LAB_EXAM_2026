import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
X, y = make_moons(n_samples=1000, noise=0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = Sequential([
    Dense(5, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

model.evaluate(X_test, y_test)


# BEGIN

# GENERATE dataset using make_moons

# SPLIT into train and test

# SCALE data

# CREATE model:
#     Dense (ReLU)
#     Dense (Sigmoid)

# COMPILE model

# TRAIN model

# EVALUATE model

# END



# // transfer learning

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Load dataset
(train_ds, val_ds) = tfds.load("tf_flowers", split=["train[:80%]", "train[80%:]"], as_supervised=True)

# Resize
def preprocess(img, label):
    img = tf.image.resize(img, (224,224))
    return img, label

train_ds = train_ds.map(preprocess).batch(32)
val_ds = val_ds.map(preprocess).batch(32)

# Base model
base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

# Model
model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_ds, epochs=3)

# Fine-tune
base.trainable = True
model.fit(train_ds, epochs=3)

model.evaluate(val_ds)




# 

# BEGIN

# LOAD dataset

# RESIZE images

# LOAD pretrained ResNet

# FREEZE base layers

# ADD new Dense layers

# COMPILE model

# TRAIN model

# UNFREEZE some layers

# FINE-TUNE model

# EVALUATE model

# END


