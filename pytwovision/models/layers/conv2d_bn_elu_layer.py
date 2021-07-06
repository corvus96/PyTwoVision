import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import MaxPooling2D

class Conv2dBNEluLayer(tf.keras.Model):
    """ A resnet block with depthwise separable convolutions to reduce the computational demand.

    Attributes: 
        name: A string that assign model name.
        num_filters: A list with a length of 2 where num_filters[0] is  the depth for the
        first depthwise separable convolutional layer and  num_filters[1] is the depth for 
        the second layer and the number of filters for the skip layer.
        stride: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width.
        dilation_rate: An integer or tuple/list of 2 integers, specifying 
        the dilation rate to use for dilated convolution.  
        Currently, specifying any `dilation_rate` value != 1 is  
        incompatible with specifying any `strides` value != 1.
    """
    def __init__(self, name, filters=32, kernel_size=3, strides=1, use_maxpool=True, postfix=None):
        super().__init__(name=name)
        
        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,  kernel_initializer='he_normal', name='conv' + postfix, padding='same')
        self.bn = BatchNormalization(name="bn" + postfix)
        self.activation = ELU(name='elu' + postfix)

        if use_maxpool:
            self.maxpool = MaxPooling2D(name='pool' + postfix)
    
    def call(self, inputs, train_batch_norm=False):
        x = self.conv2d(inputs)
        x = self.bn(x, training=train_batch_norm)
        x = self.activation(x)
        
        if hasattr(self, 'maxpool'):
            x = self.maxpool(x)
        
        return x
        
        