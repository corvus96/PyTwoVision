import tensorflow as tf
from network_builder.constants import L2_REGULARIZER_WEIGHT

class ResnetModule(tf.keras.Model):
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
    def __init__(self, name, num_filters, stride=1, dilation_rate=1):
        super().__init__(name=name)
        assert stride <= 2
        self.num_filters = num_filters
        self.stride = stride
        self.dilation_rate = dilation_rate

    def build(self, input_shapes):
        self.has_skip_conv = input_shapes[-1] != self.num_filters[-1] or self.stride > 1

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.SeparableConv2D(
            self.num_filters[0],
            (3, 3),
            padding='same',
            strides=(self.stride, self.stride),
            dilation_rate=(self.dilation_rate, self.dilation_rate),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='conv1')

        self.conv2 = tf.keras.layers.SeparableConv2D(
            self.num_filters[1],
            (3, 3),
            padding='same',
            dilation_rate=(self.dilation_rate, self.dilation_rate),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='conv2')

        if self.has_skip_conv:
            self.bn_skip = tf.keras.layers.BatchNormalization()

            self.conv_skip = tf.keras.layers.Conv2D(self.num_filters[1],
                (1, 1),
                strides=(self.stride, self.stride),
                kernel_initializer=tf.keras.initializers.he_normal(),
                kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
                name='conv_skip')

    def call(self, x, train_batch_norm=False):
        x = self.bn1(x, training=train_batch_norm)
        x = tf.keras.activations.relu(x)
        skip = x
        if self.has_skip_conv:
            skip = self.conv_skip(skip)
        x = self.conv1(x)
        x = self.bn2(x, training=train_batch_norm)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        x = x + skip
        return x