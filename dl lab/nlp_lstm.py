import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=20000)

# Padding
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test  = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# Model
model = keras.Sequential([
    layers.Embedding(20000, 100),
    layers.Bidirectional(layers.LSTM(128)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3)

# Test
model.evaluate(x_test, y_test)