import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow import keras
import argparse


# class Model(tf.keras.Model):
#     def __init__(self, n_class):
#         super(Model, self).__init__()
#         self.layer1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
#         self.layer2 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
#         self.fc = Dense(n_class, activation='relu')
#
#     def call(self, inputs, **kwargs):
#         x = self.layer1(inputs)
#         # x = MaxPooling2D(pool_size=(2, 2))(x)
#         x = self.layer2(x)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#         x = Flatten()(x)
#         x = self.fc(x)
#         return x


class Model(tf.keras.Model):
    def __init__(self, n_class):
        super(Model, self).__init__()
        self.layer1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')
        self.layer2 = Dense(100, activation='relu', kernel_initializer='he_uniform')
        self.fc = Dense(n_class, activation='relu')

    def call(self, inputs, **kwargs):
        x = self.layer1(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x
