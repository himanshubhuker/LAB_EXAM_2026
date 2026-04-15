

import numpy as np
import pandas as pd



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

xtrain = xtrain.reshape(-1, 28 * 28).astype("float32")/255.0

xtest = xtest.reshape(-1, 28 * 28).astype("float32")/255.0

numclass = 10

def MLPjourney(deep = False):
    model = keras.Sequential()
    model.add(layers.Input(shape = (784,)))
    if deep:
        model.add(layers.Dense(512, activation = "relu"))
        model.add(layers.Dense(256, activation = "relu"))
        model.add(layers.Dense(128, activation = "relu"))
    else:
        model.add(layers.Dense(512, activation = "relu"))
    model.add(layers.Dense(numclass, activation = "softmax"))
    model.compile(
        optimizer = keras.optimizers.Adam(1e-3),
        loss = "crossentropy",
        metrics = ["Accuracy"]
    )

    return model

shallow = MLPjourney(deep = False)

shallow.fit(xtrain, ytrain, epochs = 10, batch_size = 64, validation_split = 0.2)

testloss, testacc = shallow.evaluate(xtest, ytest, verbose = 1)

deep = MLPjourney(deep = True)

deep.fit(xtrain, ytrain, epochs = 10, batch_size = 64, validation_split = 0.2)

testloss, testacc = deep.evaluate(xtest, ytest, verbose = 1)

"""DEEP VS SHALLOW

Shallow
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


y_pred_probs = shallow.predict(xtest)


y_pred = np.argmax(y_pred_probs, axis=1)
y_true = ytest.flatten()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)


plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


print(classification_report(y_true, y_pred))

"""Deep"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


y_pred_probs = deep.predict(xtest)


y_pred = np.argmax(y_pred_probs, axis=1)
y_true = ytest.flatten()


cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)


plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


print(classification_report(y_true, y_pred))