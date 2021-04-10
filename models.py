from prepare_stanford import colorPreprocessingLayer, STANFORD_NO_CLASSES, IMAGE_SHAPE
from optical_flow import TVHI_NO_CLASSES, TVHI_FLOW_SHAPE
from utils import CyclicalLearningRateLogger

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Conv3D,
    MaxPooling2D,
    MaxPooling3D,
    Dropout,
    AveragePooling2D,
    TimeDistributed,
    Reshape,
    concatenate,
)
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

#you need to do:
#pip install tensorflow-addons
from tensorflow_addons.optimizers import CyclicalLearningRate

def cyclicalLRate():
    cyclical_learning_rate = CyclicalLearningRate(
     initial_learning_rate=3e-7,
     maximal_learning_rate=0.01,
     step_size=1,
     scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
     scale_mode='cycle')

    return cyclical_learning_rate

# function for creating a naive inception block
def inception_module(layer_in, f1, f2, f3):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out




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
        optimizer=Adam(learning_rate=cyclicalLRate()),
        metrics=["accuracy"],
    )

    return ("Stanford", model)


def transferModel(stanfordModel):
    prediction_layer = Dense(TVHI_NO_CLASSES, activation="softmax")

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = stanfordModel(inputs, training=True)
    x = Dense(STANFORD_NO_CLASSES)(x)
    x = Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = Model(inputs, outputs, name="Transfer")

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    return ("Transfer", model)


def opticalFlowModel():
    inputs = tf.keras.Input(shape=TVHI_FLOW_SHAPE)
    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu"))(
        inputs
    )  # Needs this due to time in the flow
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(
        x
    )  # Needs this due to time in the flow
    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu"))(
        x
    )  # Needs this due to time in the flow
    x = Conv3D(8, kernel_size=(3, 3, 3))(
        x
    )  # Maybe only use time distribution and then maxpooling 3d?
    x = MaxPooling3D(pool_size=(8, 1, 1))(x)  # Way of reducing dimentionality
    # print(x.shape) # use this to debug dimentionality. It needs to be (None, 1, W, H, D)
    x = Reshape(x.shape[2:])(x)  # (None, 1, 122, 122, 64) -> (None, 122, 122, 64)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(TVHI_NO_CLASSES, activation="softmax")(x)
    model = Model(inputs, outputs, name="Optical_Flow")

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    return ("Optical_Flow", model)


def twoStreamsModel(oneModel, flowModel):

    combinedInput = concatenate([oneModel.output, flowModel.output])
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation="relu")(combinedInput)
    output = Dense(TVHI_NO_CLASSES, activation="softmax")(x)
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the house)
    model = Model(
        inputs=[oneModel.input, flowModel.input], outputs=output, name="Two_Stream"
    )

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    return ("Two_Stream", model)
