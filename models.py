from prepare_stanford import (
    colorPreprocessingLayer,
    STANFORD_NO_CLASSES,
    IMAGE_SHAPE,
    BATCH_SIZE,
)
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

# you need to do:
# pip install tensorflow-addons
from tensorflow_addons.optimizers import CyclicalLearningRate


def cyclicalLRate(
    initial_learning_rate=3e-7,
    maximal_learning_rate=3e-5,
    step_size=1130,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
):
    cyclical_learning_rate = CyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size,
        scale_fn=scale_fn,
        scale_mode="cycle",
    )

    return cyclical_learning_rate


# function for creating a naive inception block
def inception_module(layer_in, f1, f2, f3):
    # 1x1 conv
    conv1 = Conv2D(f1, (1, 1), padding="same", activation="relu")(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2, (3, 3), padding="same", activation="relu")(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f3, (5, 5), padding="same", activation="relu")(layer_in)
    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def stanfordModel():
    """model = Sequential(name="Stanford")
    model.add(colorPreprocessingLayer())
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(STANFORD_NO_CLASSES, activation="sigmoid"))
    
    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=cyclicalLRate()),
        metrics=["accuracy"],
    )"""
    
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = colorPreprocessingLayer()(inputs, training=True)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = inception_module(x, 96, 128, 128, 32, 64, 64)
    x = Dropout(0.5)(x)
    x = inception_module(x, 96, 128, 128, 32, 64, 64)
    x = Dropout(0.5)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(256, kernel_size=(2, 2), activation="relu")(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Flatten()(x)
    
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(STANFORD_NO_CLASSES, activation="softmax")(x)
    
    model = Model(inputs, outputs, name="Stanford")
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=cyclicalLRate(maximal_learning_rate=1e-4)),
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
        optimizer=Adam(
            learning_rate=cyclicalLRate(
                maximal_learning_rate=3e-6, step_size=5 * BATCH_SIZE
            )
        ),
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
    x = TimeDistributed(Dropout(0.5))(x)  # Needs this due to time in the flow
    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu"))(
        x
    )  # Needs this due to time in the flow
    x = MaxPooling3D(pool_size=(x.shape[1], 1, 1))(x)  # Way of reducing dimentionality
    # print(x.shape) # use this to debug dimentionality. It needs to be (None, 1, W, H, D)
    x = Reshape(x.shape[2:])(x)  # (None, 1, 122, 122, 64) -> (None, 122, 122, 64)
    x = Dropout(0.5)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(TVHI_NO_CLASSES, activation="softmax")(x)
    model = Model(inputs, outputs, name="Optical_Flow")

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(
            learning_rate=cyclicalLRate(
                maximal_learning_rate=1e-2, step_size=6 * BATCH_SIZE
            )
        ),
        metrics=["accuracy"],
    )

    return ("Optical_Flow", model)


def twoStreamsModel(oneModel, flowModel):

    combinedInput = concatenate([oneModel.output, flowModel.output])
    x = Dense(4, activation="relu")(combinedInput)
    output = Dense(TVHI_NO_CLASSES, activation="softmax")(x)
    model = Model(
        inputs=[oneModel.input, flowModel.input], outputs=output, name="Two_Stream"
    )

    # Compile for training
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(
            learning_rate=cyclicalLRate(
                maximal_learning_rate=3e-4, step_size=5 * BATCH_SIZE
            )
        ),
        metrics=["accuracy"],
    )

    return ("Two_Stream", model)


def hydraModel():
    name = "Hydra"

    def stanfordBody():
        inputs = tf.keras.Input(shape=IMAGE_SHAPE)
        x = colorPreprocessingLayer()(inputs)
        x = Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        output = MaxPooling2D(pool_size=(3, 3))(x)

        model = Model(inputs=inputs, outputs=output, name="Stanford-Body")

        return model

    def stanfordHeadModel(body):
        x = Flatten()(body.output)
        x = Dense(100, activation="relu")(x)
        output = Dense(STANFORD_NO_CLASSES, activation="softmax")(x)

        model = Model(inputs=body.input, outputs=output, name="Stanford-Head")

        # Compile for training
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(learning_rate=cyclicalLRate()),
            metrics=["accuracy"],
        )

        return ("Stanford-Head", model)

    def flowBody():
        inputs = tf.keras.Input(shape=TVHI_FLOW_SHAPE)
        x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu"))(
            inputs
        )  # Needs this due to time in the flow
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(
            x
        )  # Needs this due to time in the flow
        x = TimeDistributed(Dropout(0.5))(x)  # Needs this due to time in the flow
        x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu"))(
            x
        )  # Needs this due to time in the flow
        x = MaxPooling3D(pool_size=(x.shape[1], 1, 1))(
            x
        )  # Way of reducing dimentionality
        # print(x.shape) # use this to debug dimentionality. It needs to be (None, 1, W, H, D)
        x = Reshape(x.shape[2:])(x)  # (None, 1, 122, 122, 64) -> (None, 122, 122, 64)
        x = Dropout(0.5)(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        output = MaxPooling2D(pool_size=(3, 3), padding="same")(x)

        model = Model(inputs=inputs, outputs=output, name="Flow-Body")
        return model

    def transferFusion(stanford, flow):
        output = concatenate([stanford_body.output, flow_body.output])

        model = Model(
            inputs=[stanford.input, flow.input], outputs=output, name="Transfer-Fusion"
        )

        return model

    def flowFusion(stanford, flow):
        output = concatenate([stanford_body.output, flow_body.output])

        model = Model(
            inputs=[stanford.input, flow.input], outputs=output, name="Flow-Fusion"
        )

        return model

    def transferHead(body):
        x = Dropout(0.5)(body.output)
        x = Conv2D(256, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(100, activation="relu")(x)
        output = Dense(TVHI_NO_CLASSES, activation="sigmoid")(x)

        model = Model(inputs=body.input, outputs=output, name="Transfer-Head")

        # Compile for training
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(
                learning_rate=cyclicalLRate(
                    maximal_learning_rate=3e-6, step_size=5 * BATCH_SIZE
                )
            ),
            metrics=["accuracy"],
        )

        return model

    def flowHead(body):
        x = Dropout(0.5)(body.output)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(TVHI_NO_CLASSES, activation="sigmoid")(x)

        model = Model(body.input, outputs, name="Flow_Head")

        # Compile for training
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(
                learning_rate=cyclicalLRate(
                    maximal_learning_rate=1e-2, step_size=6 * BATCH_SIZE
                )
            ),
            metrics=["accuracy"],
        )

        return model

    def hydraHeadModel(transferFusedHead, flowFusedHead):
        combinedInput = concatenate([transferFusedHead.output, flowFusedHead.output])
        x = Dense(32, activation="relu")(combinedInput)
        output = Dense(TVHI_NO_CLASSES, activation="softmax")(x)
        model = Model(inputs=transferFusedHead.input, outputs=output, name=name)

        # Compile for training
        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(
                learning_rate=cyclicalLRate(
                    maximal_learning_rate=3e-3, step_size=3 * BATCH_SIZE
                )
            ),
            metrics=["accuracy"],
        )

        return (name, model)

    stanford_body = stanfordBody()
    stanford_model = stanfordHeadModel(stanford_body)

    flow_body = flowBody()

    transfer_fusion = transferFusion(stanford_body, flow_body)
    flow_fusion = flowFusion(stanford_body, flow_body)

    transfer_head = transferHead(transfer_fusion)
    flow_head = flowHead(flow_fusion)

    hydra_model = hydraHeadModel(transfer_head, flow_head)

    return stanford_model, hydra_model