import tensorflow as tf

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import conv2d_bn_leaky_relu_layer

def residual_layer(input_layer, input_channel, filter_num1, filter_num2):
    """ A residual block with two Convolutional-batch normalization-leakyReLu layers stacked one above other.

    Args: 
        input_layer: A tensor that works like input.
        input_channel: input layer dimensions.
        filter_num1: filter depth for the first convolutional-batch normalization-leakyRelu layer.
        filter_num2: filter depth for the second convolutional-batch normalization-leakyRelu layer.

    Returns:
        A residual block.
    """
    short_cut = input_layer
    x = conv2d_bn_leaky_relu_layer(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    x = conv2d_bn_leaky_relu_layer(x, filters_shape=(3, 3, filter_num1,   filter_num2))
    residual_output = short_cut + x

    return residual_output