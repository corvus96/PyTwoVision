import tensorflow as tf
from models.constants import L2_REGULARIZER_WEIGHT

class Upsample(tf.keras.Model):
    """ A stack that unsample by a factor to compensate downsampling.
    The stack is formed by a batch normalization layer, next a transpose Conv2D layer, and
    finally a relu layer

    Attributes: 
        name: A string that assign model name.
        factor: An integer that changes the kernel size of transpose conv2D layer. The kernel size
        is kernel_size = 2 * factor - factor % 2.
        num_output_channels: Integer, the dimensionality of the output space  
        (i.e. the number of output filters in the convolution).
    """
    def __init__(self, name, factor, num_output_channels):
        super().__init__(name=name)

        self.bn = tf.keras.layers.BatchNormalization()

        kernel_size = 2 * factor - factor % 2
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(num_output_channels,
            (kernel_size, kernel_size),
            (factor, factor),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='transposed_conv')

    def call(self, x, train_batch_norm=False):
        x = self.bn(x, training=train_batch_norm)
        x = self.transposed_conv(x)
        x = tf.nn.relu(x)
        return x