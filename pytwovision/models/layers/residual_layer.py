import tensorflow as tf

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import Conv2dBNLeakyReluLayer

class ResidualLayer(tf.keras.Model):
    """ A residual block with two Convolutional-batch normalization-leakyReLu layers stacked one above other.

    Attributes: 
        input_channel: input layer dimensions
        filter_num1: filter depth for the first convolutional-batch normalization-leakyRelu layer.
        filter_num2: filter depth for the second convolutional-batch normalization-leakyRelu layer.
        postfix: an unique id to identifier each layer in a model
    """
    def __init__(self, input_channel, filter_num1, filter_num2):
        super().__init__()
        self.conv1 = Conv2dBNLeakyReluLayer(filters_shape=(1, 1, input_channel, filter_num1))
        self.conv2 = Conv2dBNLeakyReluLayer(filters_shape=(3, 3, filter_num1,   filter_num2))
    def call(self, inputs):
        short_cut = inputs
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = short_cut + x
        return x
        