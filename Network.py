from keras import Input, Model
from keras.layers import Activation, Add, BatchNormalizationV2, Conv2D, Dense, Flatten
import tensorflow as tf


class Network(object):

    def __init__(self, is_uniform=False):
        if is_uniform:
            return

        inputs = Input(shape=(8, 8, 16))

        # convolutional block
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(inputs)
        x = BatchNormalizationV2()(x)
        x = Activation(activation='relu')(x)

        for k in range(19):
            # residual block
            residual_block_input = x
            x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalizationV2()(x)
            x = Activation(activation='relu')(x)
            x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalizationV2()(x)
            x = Add()([x, residual_block_input])
            x = Activation(activation='relu')(x)

        # policy head
        policy_x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        policy_x = BatchNormalizationV2()(policy_x)
        policy_x = Activation(activation='relu')(policy_x)
        policy_x = Conv2D(filters=73, kernel_size=(3, 3), padding='same')(policy_x)
        policy_x = Activation(activation='softmax')(policy_x)
        policy_outputs = Flatten()(policy_x)

        # value head
        value_x = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x)
        value_x = BatchNormalizationV2()(value_x)
        value_x = Activation(activation='relu')(value_x)
        value_x = Flatten()(value_x)
        value_x = Dense(units=256)(value_x)
        value_x = Activation(activation='relu')(value_x)
        value_x = Dense(units=1)(value_x)
        value_outputs = Activation(activation='tanh')(value_x)

        self.model = Model(inputs, [value_outputs, policy_outputs])

    @tf.function
    def inference(self, image):
        return self.model(image)  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return self.model.trainable_weights
