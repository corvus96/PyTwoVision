import tensorflow as tf

class UpsampleLayer(tf.keras.layers.Layer):
    """ A layer for upsampling an image. """
    def call(self, inputs):
        return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')

        