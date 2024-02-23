import keras
from keras import layers
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import tensorflow as tf
import random
import datetime
import math


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=(*img_size, 3))
    x = layers.Conv2D(32, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # list for storing skip connections of U-NET
    prevli = [x]
    i = 0
    i += 1
    prev_block_activation = x

    for filters in [32, 64]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding='same')(
            prev_block_activation
        )

        x = layers.add([residual, x])
        prev_block_activation = x

        prevli += [x]
        i += 1

    for filters in [64, 32, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        i -= 1
        x = layers.add([layers.UpSampling2D(2)(prevli[i]), x])


    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(outputs)
    model = keras.Model(inputs, outputs)
    return model
