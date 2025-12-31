import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

def tworzenie_modelu_2_warstwy():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)),
      tf.keras.layers.Rescaling(1./255),

      tf.keras.layers.Conv2D(32, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(64, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Flatten(), 
      tf.keras.layers.Dense(256), 
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dense(10), 
      tf.keras.layers.Activation("softmax")
  ])
  return model

def tworzenie_modelu_3_warstwy():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)),
      tf.keras.layers.Rescaling(1./255),

      tf.keras.layers.Conv2D(32, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(64, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(128, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256), 
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dense(10), 
      tf.keras.layers.Activation("softmax")
  ])
  return model

def tworzenie_modelu_3_warstwy_dropout_15():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)),
      tf.keras.layers.Rescaling(1./255),

      tf.keras.layers.Conv2D(32, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(64, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(128, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256), 
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dropout(0.15),
      tf.keras.layers.Dense(10), 
      tf.keras.layers.Activation("softmax")
  ])
  return model

def tworzenie_modelu_3_warstwy_dropout_25():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)),
      tf.keras.layers.Rescaling(1./255),

      tf.keras.layers.Conv2D(32, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(64, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(128, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256), 
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(10), 
      tf.keras.layers.Activation("softmax")
  ])
  return model
  
def tworzenie_modelu_3_warstwy_dropout_50():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)),
      tf.keras.layers.Rescaling(1./255),

      tf.keras.layers.Conv2D(32, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(64, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(128, kernel_size = 3),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256), 
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dropout(0.50),
      tf.keras.layers.Dense(10), 
      tf.keras.layers.Activation("softmax")
  ])
  return model

def tworzenie_modelu_3_warstwy_dropout_25_batch_normalization():
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(32, 32, 3)),
      tf.keras.layers.Rescaling(1./255),

      tf.keras.layers.Conv2D(32, kernel_size = 3),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(64, kernel_size = 3),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Conv2D(128, kernel_size = 3),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256), 
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(10), 
      tf.keras.layers.Activation("softmax")
  ])
  return model

def modelowanie(model, epoki=20):

    model.compile(
      optimizer=keras.optimizers.Adam(3e-4),
      loss=keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=epoki,
        batch_size=64,
        validation_data=(x_test, y_test)
    )

    return history

modelowanie(tworzenie_modelu_2_warstwy())
modelowanie(tworzenie_modelu_3_warstwy())
modelowanie(tworzenie_modelu_3_warstwy_dropout_15())
modelowanie(tworzenie_modelu_3_warstwy_dropout_25())
modelowanie(tworzenie_modelu_3_warstwy_dropout_50())
modelowanie(tworzenie_modelu_3_warstwy_dropout_25_batch_normalization())
