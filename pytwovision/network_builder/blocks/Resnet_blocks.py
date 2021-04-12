import tensorflow as tf
from network_builder.constants import L2_REGULARIZER_WEIGHT
from network_builder.layers.Resnet import ResnetModule

class ResnetBackbone(tf.keras.Model):
    """ Create a network block for simultaneous semantic segmentation 
        and object detection, the backbone model is based on ResNet-38. 
        
        The output of of the backbone is then fed into
        multiple network heads. One head is the semantic segmentation head
        The second head is the object detection head.
        
        Attributes: 
            name: A string that assign model name
            pretrain: A boolean, if is true resnet_module_3a, resnet_module_3b, resnet_module_3c 
            will have an stride 2 and a dilation_rate of 1, else stride 1 and a dilation_rate of 2. Also in case is true resnet_module_3d, resnet_module_3e, resnet_module_3f will have an stride 1 and a dilation_rate of 4, 8, 4 respectively, else stride 1 and a dilation_rate of 1, 2, 1 respectively
        """
    def __init__(self, name, pretrain=False):
        super().__init__(name=name)

        self.first_conv = tf.keras.layers.Conv2D(64,
            (7, 7),
            padding='same',
            strides=(2, 2),
            kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZER_WEIGHT),
            name='first_conv')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPooling2D(2)
        self.maxpool2 = tf.keras.layers.MaxPooling2D(2)

        self.module_1a = ResnetModule('resnet_module_1a', [64, 64])
        self.module_1b = ResnetModule('resnet_module_1b', [64, 64])
        self.module_1c = ResnetModule('resnet_module_1c', [64, 64])

        self.module_2a = ResnetModule('resnet_module_2a', [128, 128])
        self.module_2b = ResnetModule('resnet_module_2b', [128, 128])
        self.module_2c = ResnetModule('resnet_module_2c', [128, 128])
        self.module_2d = ResnetModule('resnet_module_2d', [128, 128])

        stride = 1
        dilation_rate = 1
        if pretrain:
            stride = 2
        else:
            dilation_rate = 2
        self.module_3a = ResnetModule('resnet_module_3a', [256, 256], stride=stride,
                                      dilation_rate=dilation_rate)
        self.module_3b = ResnetModule('resnet_module_3b', [256, 256],
                                      dilation_rate=dilation_rate)
        self.module_3c = ResnetModule('resnet_module_3c', [256, 256],
                                      dilation_rate=dilation_rate)

        if not pretrain:
            dilation_rate = 4
        self.module_3d = ResnetModule('resnet_module_3d', [512, 512], stride=stride,
                                      dilation_rate=dilation_rate)
        self.module_3e = ResnetModule('resnet_module_3e', [512, 512],
                                      dilation_rate=2*dilation_rate)
        self.module_3f = ResnetModule('resnet_module_3f', [512, 512],
                                      dilation_rate=dilation_rate)

    def call(self, x, train_batch_norm=False):
        x = self.first_conv(x)
        x = tf.keras.activations.relu(x)
        x = self.bn1(x, training=train_batch_norm)

        x = self.module_1a(x, train_batch_norm=train_batch_norm)
        x = self.module_1b(x, train_batch_norm=train_batch_norm)

        x = self.maxpool1(x)

        x = self.module_1c(x, train_batch_norm=train_batch_norm)
        x = self.module_2a(x, train_batch_norm=train_batch_norm)

        x = self.maxpool2(x)

        x = self.module_2b(x, train_batch_norm=train_batch_norm)
        x = self.module_2c(x, train_batch_norm=train_batch_norm)
        x = self.module_2d(x, train_batch_norm=train_batch_norm)

        x = self.module_3a(x, train_batch_norm=train_batch_norm)
        x = self.module_3b(x, train_batch_norm=train_batch_norm)
        x = self.module_3c(x, train_batch_norm=train_batch_norm)

        x = self.module_3d(x, train_batch_norm=train_batch_norm)
        x = self.module_3e(x, train_batch_norm=train_batch_norm)
        x = self.module_3f(x, train_batch_norm=train_batch_norm)
        return x


class ResnetBranch(tf.keras.Model):
    """ Create a commom block that can be used in Simultaneous 
    Object Detection and Semantic Segmentation network especifically in 
    semantic segmentation head and object detection head.
    
    Attributes: 
            name: A string that assign model name
    """
    def __init__(self, name):
        super().__init__(name=name)

        self.module_4a = ResnetModule('resnet_module_4a', [512, 512])
        self.module_4b = ResnetModule('resnet_module_4b', [512, 512])
        self.module_4c = ResnetModule('resnet_module_4c', [512, 512])

    def call(self, x, train_batch_norm=False):
        x = self.module_4a(x, train_batch_norm=train_batch_norm)
        x = self.module_4b(x, train_batch_norm=train_batch_norm)
        x = self.module_4c(x, train_batch_norm=train_batch_norm)
        return x