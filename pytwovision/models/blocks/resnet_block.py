"""ResNet model builder as SSD backbone
Adopted fr Chapter 2 of ADL - Deep Networks
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""
from abc import ABC, abstractmethod

from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model

from models.layers.conv2d_bn_elu_layer import Conv2dBNEluLayer
from models.layers.conv2d_bn_relu_layer import Conv2dBNReluLayer

class ResnetStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def build(self):
        pass

class ResnetBlock():

    def __init__(self, strategy: ResnetStrategy):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> ResnetStrategy:
        """
        The ResnetBlock maintains a reference to one of the Strategy objects. The
        ResnetBlock does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: ResnetStrategy):
        """
        To replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def build_model(self, input_shape, n_layers=4, n=6):
        """Build a resnet as backbone of SSD
        # Arguments:
            input_shape (list): Input image size and channels
            n_layers (int): Number of feature layers for SSD
            version (int): Supports ResNetv1 and v2
            n (int): Determines number of ResNet layers
                    (Default is ResNet50)
        # Returns
            model (Keras Model)
        """
        if type(self._strategy).__name__ == 'V1':
            depth = n * 6 + 2
        elif type(self._strategy).__name__ == 'V2':
            depth = n * 9 + 2
        return self._strategy.build(input_shape=input_shape, depth=depth, n_layers=n_layers)

class V1(ResnetStrategy):
    def build(self, input_shape, depth, n_layers=10):
        """ResNet Version 1 Model builder [a]
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M
        # Arguments
            input_shape (tensor): Shape of input image tensor
            depth (int): Number of core convolutional layers
            num_classes (int): Number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        conv_layer_initial = Conv2dBNReluLayer("Feature layer initial")
        x = conv_layer_initial(inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                conv_layer_A = Conv2dBNReluLayer("Feature layer type A", num_filters=num_filters, strides=strides)
                y = conv_layer_A(x)
                conv_layer_B = Conv2dBNReluLayer("Feature layer type B", num_filters=num_filters, activation=None)
                y = conv_layer_B(y)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    conv_layer_C = Conv2dBNReluLayer("Feature layer type C", num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                    x = conv_layer_C(x)
                x = Add()([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # 1st feature map layer
        conv = AveragePooling2D(pool_size=4, name='pool1')(x)

        outputs = [conv]
        prev_conv = conv
        n_filters = 64

        # additional feature map layers
        for i in range(n_layers - 1):
            postfix = "_layer" + str(i+2)
            conv = Conv2dBNEluLayer("Aditional Featured layer " + str(i + 1), n_filters, kernel_size=3, strides=2, use_maxpool=False, postfix=postfix)
            conv = conv(prev_conv)
            outputs.append(conv)
            prev_conv = conv
            n_filters *= 2
        

        # instantiate model
        name = 'ResNet%dv1' % (depth)
        model = Model(inputs=inputs,
                    outputs=outputs,
                    name=name)
        return model

class V2(ResnetStrategy):
    def build(self, input_shape, depth, n_layers=4):
        """ResNet Version 2 Model builder
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
        # Arguments
            input_shape (tensor): Shape of input image tensor
            depth (int): Number of core convolutional layers
            num_classes (int): Number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        conv_layer_initial = Conv2dBNReluLayer("Feature layer initial", num_filters=num_filters_in,
                        conv_first=True)
        x = conv_layer_initial(inputs)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                conv_layer_A = Conv2dBNReluLayer("Feature layer type A", num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)
                y = conv_layer_A(x)
                conv_layer_B = Conv2dBNReluLayer("Feature layer type B", num_filters=num_filters_in,
                                conv_first=False)
                y = conv_layer_B(y)
                conv_layer_C = Conv2dBNReluLayer("Feature layer type C", num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
                y = conv_layer_C(y)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    conv_layer_D = Conv2dBNReluLayer("Feature layer type D", num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                    x = conv_layer_D(x)
                x = Add()([x, y])

            num_filters_in = num_filters_out

        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # 1st feature map layer
        conv = AveragePooling2D(pool_size=4, name='pool1')(x)
        outputs = [conv]
        prev_conv = conv
        n_filters = 64

        # additional feature map layers
        for i in range(n_layers - 1):
            postfix = "_layer" + str(i+2)
            conv = Conv2dBNEluLayer("Aditional Featured layer " + str(i + 1), n_filters, kernel_size=3, strides=2, use_maxpool=False, postfix=postfix)
            conv = conv(prev_conv)
            outputs.append(conv)
            prev_conv = conv
            n_filters *= 2
        

        # instantiate model.
        name = "ResNet{depth}v2".format(depth=depth)
        model = Model(inputs=inputs,
                    outputs=outputs,
                    name=name)
        return model