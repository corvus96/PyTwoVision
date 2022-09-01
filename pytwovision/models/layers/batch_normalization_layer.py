import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

class BatchNormalization(BatchNormalization):
    """ A modified batch normalization layer.

    Args:
        x: input of the layer.
        training: it's necessary to know when freeze the layer weights.

    Returns:
        A batchNormalization layer.
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)