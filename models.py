from prepare_stanford import colorPreprocessingLayer, STANFORD_NO_CLASSES, IMAGE_SHAPE
from optical_flow import TVHI_NO_CLASSES

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    AveragePooling2D,
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def stanfordModel():
    model = Sequential(name="Stanford")
    model.add(colorPreprocessingLayer())
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(STANFORD_NO_CLASSES, activation="softmax"))

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    return ("Stanford", model)


def transferModel(stanfordModel):
    prediction_layer = tf.keras.layers.Dense(TVHI_NO_CLASSES, activation="softmax")

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = stanfordModel(inputs, training=True)
    x = tf.keras.layers.Dense(STANFORD_NO_CLASSES)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs, name="Transfer")

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    return ("Transfer", model)


def opticalFlowModel():
    return ("Optical_Flow", None)


def twoStreamsModel(oneModel, flowModel):
    return ("Two_Stream", None)