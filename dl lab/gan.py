import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 100

# Generator
generator = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
    layers.Dense(784, activation='tanh'),
    layers.Reshape((28,28,1))
])

# Discriminator
discriminator = keras.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN
discriminator.trainable = False

gan = keras.Sequential([generator, discriminator])

gan.compile(optimizer='adam', loss='binary_crossentropy')

# Load data
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_train = x_train[..., None]



#

# INITIALIZE Generator

# INITIALIZE Discriminator

# COMBINE into GAN

# FOR each epoch:
#     GENERATE fake images

#     TRAIN Discriminator on:
#         real images
#         fake images

#     TRAIN Generator to fool Discriminator

# END FOR

# GENERATE new images

# END