import keras
from keras import layers
import tensorflow as tf


class Network(object):

    def __init__(self):
        inputs = keras.Input(shape=(8, 8, 16))

        # convolutional block
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(inputs)
        x = layers.BatchNormalizationV2()(x)
        x = layers.Activation(activation='relu')(x)

        for k in range(19):
            # residual block
            residual_block_input = x
            x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalizationV2()(x)
            x = layers.Activation(activation='relu')(x)
            x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
            x = layers.BatchNormalizationV2()(x)
            x = layers.Add()([x, residual_block_input])
            x = layers.Activation(activation='relu')(x)

        # policy head
        policy_x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        policy_x = layers.BatchNormalizationV2()(policy_x)
        policy_x = layers.Activation(activation='relu')(policy_x)
        policy_x = layers.Conv2D(filters=73, kernel_size=(3, 3), padding='same')(policy_x)
        policy_x = layers.Activation(activation='softmax')(policy_x)
        policy_outputs = layers.Flatten()(policy_x)

        # value head
        value_x = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x)
        value_x = layers.BatchNormalizationV2()(value_x)
        value_x = layers.Activation(activation='relu')(value_x)
        value_x = layers.Flatten()(value_x)
        value_x = layers.Dense(units=256)(value_x)
        value_x = layers.Activation(activation='relu')(value_x)
        value_x = layers.Dense(units=1)(value_x)
        value_outputs = layers.Activation(activation='tanh')(value_x)

        self.model = keras.Model(inputs, [value_outputs, policy_outputs])

    @tf.function
    def inference(self, image):
        value, policy = self.model(image)
        value = tf.reshape(value, shape=())
        policy = tf.squeeze(policy, axis=[0])
        return value, policy  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return self.model.get_weights()
