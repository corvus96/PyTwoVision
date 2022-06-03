import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.regularizers import l2

from pytwovision.models.layers.batch_normalization_layer import BatchNormalization

class Conv2dBNLeakyReluLayer(tf.keras.Model):
    """ A resnet block with depthwise separable convolutions to reduce the computational demand.

    Attributes: 
        name: A string that assign model name.
        filters_shape: An array or list or tuple that contains the shape of 
        the filters which filters_shape[0] is kernel size for conv2d layer.
        down_sample: a boolean to know when apply zero padding layer.
        activate: a boolean to know when apply leaky ReLu layer.
        bn: a boolean to know when apply a batch normalization layer.
        postfix: an unique id to identifier each layer in a model
    """
    def __init__(self, name, filters_shape, down_sample=False, activate=True, bn=True, postfix=None):
        super().__init__(name=name)
        if type(postfix) == int or type(postfix) == float:
            postfix = str(postfix)
        if type(postfix) is not str:
            raise Exception('postfix has to be a string')
        if down_sample:
            self.down_sample_layer = ZeroPadding2D(((1, 0), (1, 0)))
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        self.conv2d = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], name= name + postfix, strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))
        if bn:
            self.bn = BatchNormalization()
        
        if activate == True:
            self.activation = LeakyReLU(alpha=0.1, name='LeakyRelu' + postfix)

    
    def call(self, inputs):
        if hasattr(self, 'down_sample_layer'):
            inputs = self.down_sample_layer(inputs)
        x = self.conv2d(inputs)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        
        return x
        