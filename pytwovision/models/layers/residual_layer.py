import tensorflow as tf

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import Conv2dBNLeakyReluLayer

class ResidualLayer(tf.keras.Model):
    """ A residual block with two Convolutional-batch normalization-leakyReLu layers stacked one above other.

    Attributes: 
        name: A string that assign model name.
        input_channel: input layer dimensions
        filter_num1: filter depth for the first convolutional-batch normalization-leakyRelu layer.
        filter_num2: filter depth for the second convolutional-batch normalization-leakyRelu layer.
        postfix: an unique id to identifier each layer in a model
    """
    def __init__(self, name, input_channel, filter_num1, filter_num2, postfix=None):
        super().__init__(name=name)
        if type(postfix) == int or type(postfix) == float:
            postfix = str(postfix)
        if type(postfix) is not str:
            raise Exception('postfix has to be a string')
        self.conv1 = Conv2dBNLeakyReluLayer(name + "_conv_layer_A_" + postfix, filters_shape=(1, 1, input_channel, filter_num1), postfix=postfix)
        self.conv2 = Conv2dBNLeakyReluLayer(name + "_conv_layer_B_" + postfix, filters_shape=(3, 3, filter_num1,   filter_num2), postfix=postfix)
    def call(self, inputs):
        short_cut = inputs
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = short_cut + x
        return x
        