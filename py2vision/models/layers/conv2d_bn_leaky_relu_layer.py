import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.regularizers import l2

from py2vision.models.layers.batch_normalization_layer import BatchNormalization

def conv2d_bn_leaky_relu_layer(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    """ A resnet block with depthwise separable convolutions to reduce the computational demand.

    Args: 
        input_layer: A tensor that works like input.
        filters_shape: An array or list or tuple that contains the shape of the filters which filters_shape[0] is kernel size for conv2d layer.
        downsample: a boolean to know when apply zero padding layer.
        activate: a boolean to know when apply leaky ReLu layer.
        bn: a boolean to know when apply a batch normalization layer.

    Returns: 
        A resnet block
    """
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    x = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        x = BatchNormalization()(x)
    if activate == True:
        x = LeakyReLU(alpha=0.1)(x)

    return x