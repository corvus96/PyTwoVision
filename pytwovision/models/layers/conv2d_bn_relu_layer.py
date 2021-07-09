import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

class Conv2dBNReluLayer(tf.keras.Model):
    """Build a resnet as backbone of SSD
    # Arguments:
        input_shape (list): Input image size and channels
        n_layers (int): Number of feature layers for SSD
        version (int): Supports ResNetv1 and v2 but v2 by default
        n (int): Determines number of ResNet layers
                 (Default is ResNet50)
    # Returns
        model (Keras Model)
    """
    def __init__(self, name, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
        super().__init__(name=name)
        self.conv2d = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))     
        
        self.conv_first = conv_first
        if batch_normalization:
            self.bn = BatchNormalization()
        if activation is not None:
            self.activation = Activation(activation)
    
    def call(self, inputs, train_batch_norm=False):
        if self.conv_first is True:
            x = self.conv2d(inputs)
            if hasattr(self, 'bn'):
                x = self.bn(x, training=train_batch_norm)
            if hasattr(self, 'activation'):
                x = self.activation(x)
        else:
            if hasattr(self, 'bn'):
                x = self.bn(inputs, training=train_batch_norm)
            if hasattr(self, 'activation'):
                x = self.activation(x)
            x = self.conv2d(x)
        
        return x



