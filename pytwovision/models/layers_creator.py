import tensorflow as tf
from tensorflow import keras

class DownsampleLayer(keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, pad="valid",dilation=(1,1), conv_dim="2D", apply_batchnorm=True, alpha=0.2):
        """ This is a class that create a downsampleLayer, formed by 
        3 layers a convolutional layer, a batch normalization layer and 
        a LeakyReLu activation function

        Attributes:
            filters: Integer, the dimensionality of the output space (i.e. the number of  
            output filters in the convolution).
            kernel_size: An integer or tuple/list of 2 integers, specifying the height
            the same value for all spatial dimensions. 
            strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width. Can be a single integer to  
            specify the same value for all spatial dimensions. Specifying any stride  
            value != 1 is incompatible with specifying any `dilation_rate` value != 1.  
            pad: one of `"valid"` or `"same"` (case-insensitive).  
            `"valid"` means no padding. `"same"` results in padding evenly to  
            the left/right or up/down of the input such that output has the same  
            height/width dimension as the input.  
            dilation_rate: an integer or tuple/list of 2 integers, specifying the  
            dilation rate to use for dilated convolution. Can be a single integer to  
            specify the same value for all spatial dimensions. Currently, specifying  
            any `dilation_rate` value != 1 is incompatible with specifying any stride  
            value != 1.  
            conv_dim:  A string that select a type of conv layer `"2D"` for conv2D and
            `"3D"` for conv3D
            apply_batchnorm: A boolean that activate or desactive batch normalization layer
            alpha: A Float that change negative slope coefficient of LeakyReLu. Default to 0.2.
            
        """
        super(DownsampleLayer, self).__init__()
        if conv_dim == "2D":
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Conv2D(filters=filters,padding=pad, kernel_size=kernel_size, strides=stride, dilation_rate=dilation))
            
            if apply_batchnorm: 
                self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
        elif conv_dim == "3D":
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Conv3D(filters=filters, padding=pad, kernel_size=kernel_size, strides=stride))
            
            if apply_batchnorm: 
                self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.LeakyReLU(alpha=alpha))
        else:
            raise ValueError(f'conv_dim cannot be {conv_dim}, it must be either 2D or 3D.')
        
    def call(self, input):
        """ Feedforward pass of DownsampleLayer
         Args:
            input: A tensor that pass trought the model.
        Returns:
            A new tensor after con2D -> batch normalization -> LeakyReLu.
            or con3D -> batch normalization -> LeakyReLu.

        """
        out = self.model(input)
        return out
    
    def get_config(self):
        """ Layers configurations resume
        Returns: 
            A dictionary with each layer attributes

        """
        return self.model.get_config()
            