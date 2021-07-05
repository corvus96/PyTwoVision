import tensorflow as tf
from .Resnet import ResnetModule

class SingleDecoder(tf.keras.Model):
    """A decoder formed by two ResnetModules and 1 conv2D with 1x1 kernel 
       
       Attributes: 
        name: A string that assign model name.
        num_output_channels: An Integer that define output dimensionality for con2D layer.
        channels: An integer that define the depth for ResnetModules.
    """
    def __init__(self, name, num_output_channels, channels):
        super().__init__(name=name)

        self.num_output_channels = num_output_channels

        self.rm1 = ResnetModule('resnet_module_1', [channels, channels])
        self.rm2 = ResnetModule('resnet_module_2', [channels, channels])
        self.last_conv = tf.keras.layers.Conv2D(num_output_channels,
                (1, 1),
                kernel_initializer=tf.keras.initializers.he_normal(),
                name='last_conv')

    def call(self, x, train_batch_norm=False):
        n = x.get_shape().as_list()[0]
        # This is an ugly hack for saved_model. Without it fails to determine the batch size
        if n is None:
            n = 1

        x = self.rm1(x, train_batch_norm=train_batch_norm)
        x = self.rm2(x, train_batch_norm=train_batch_norm)
        x = self.last_conv(x)
        x = tf.reshape(x, [n, -1, self.num_output_channels])
        return x