

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

"""mnist dataset"""

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

num_classes = 10

from re import L
def build_small_cnn():
  model = keras.Sequential(
      [
          layers.Input(shape = (28,28,1)),
          layers.MaxPool2D((2, 2)),
          layers.Conv2D(64, 3, padding = "same" , activation="relu"),
          layers.Flatten(),
          layers.Dense(128 , activation="relu"),
          layers.Dense(num_classes , activation="softmax"),
      ])
  model.compile(
      optimizer= keras.optimizers.Adam(),
      # loss= "sparse_Categorical_crossentropy()",
      loss = keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"]
  )

  return model

model = build_small_cnn()
history = model.fit(x_train, y_train, batch_size = 64, epochs = 30, validation_split = 0.1)

text_loss , test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")



"""2nd Code cifar10 dataset

"""



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0


def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

model = build_cnn()


early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    validation_split=0.1,
    callbacks=[early_stop]
)


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred_classes)

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - CIFAR10")
plt.show()

