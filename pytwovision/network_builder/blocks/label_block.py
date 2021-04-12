import tensorflow as tf
from layers.Upsample import Upsample
from Resnet_blocks import ResnetBranch

class LabelBranch(tf.keras.Model):
    """ A semantic segmentation head block that works like a decoder. 
    After three more ResNet modules (ResnetBranch) the data tensor
    is upsampled again so that the final segmentation
    map has the same resolution as the input image.
    This is done using three transposed convolutions
    that each learn to upsample by a factor of 2. The
    final convolution layer then reduces the number of
    channels to the number of classes.
    
    Attributes: 
            name: A string that assign model name
    """
    def __init__(self, name, config):
        super().__init__(name=name)

        self.core_branch = ResnetBranch('label_branch')
        self.upsample1 = Upsample('upsample1', 2, 128)
        self.upsample2 = Upsample('upsample2', 2, 64)
        self.upsample3 = Upsample('upsample2', 2, 32)
        self.final_conv = tf.keras.layers.Conv2D(config['num_label_classes'],
            (1, 1),
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='final_conv')

    def call(self, x, train_batch_norm=False):
        x = self.core_branch(x, train_batch_norm=train_batch_norm)
        x = self.upsample1(x, train_batch_norm=train_batch_norm)
        x = self.upsample2(x, train_batch_norm=train_batch_norm)
        x = self.upsample3(x, train_batch_norm=train_batch_norm)
        x = self.final_conv(x)
        return x