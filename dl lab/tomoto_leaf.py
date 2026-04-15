# ✅ 1. EASY CODE (WRITE THIS IN EXAM)

# 👉 Only write this core version (don’t write full long code)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory("train", image_size=(224,224), batch_size=32)
val_ds   = tf.keras.preprocessing.image_dataset_from_directory("valid", image_size=(224,224), batch_size=32)
test_ds  = tf.keras.preprocessing.image_dataset_from_directory("test",  image_size=(224,224), batch_size=32)

# Data augmentation
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

# Pretrained model
base = keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(224,224,3))
base.trainable = False

# Model
model = keras.Sequential([
    data_aug,
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Test
model.evaluate(test_ds)








# 📊 2. CLASS DISTRIBUTION (SHORT)

# 👉 Write only this logic:

import numpy as np

count0, count1 = 0, 0

for _, labels in train_ds:
    labels = labels.numpy().flatten()
    count0 += sum(labels==0)
    count1 += sum(labels==1)

print(count0, count1)




# 📊 3. METRICS (SHORT)

from sklearn.metrics import classification_report
import numpy as np

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())

y_pred = model.predict(test_ds)
y_pred = (y_pred > 0.5).astype(int)

print(classification_report(y_true, y_pred))






# Transfer Learning with DenseNet

# BEGIN
# LOAD image dataset

# APPLY preprocessing

# APPLY data augmentation

# LOAD pretrained DenseNet

# FREEZE base layers

# ADD Dense layers

# COMPILE model

# TRAIN model

# EVALUATE model

# COUNT class distribution

# PREDICT test data

# CALCULATE precision, recall, F1

# END

