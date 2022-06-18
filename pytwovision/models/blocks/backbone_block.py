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
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import Conv2dBNLeakyReluLayer
from pytwovision.models.layers.residual_layer import ResidualLayer

class BackboneStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def build(self):
        pass

class BackboneBlock():

    def __init__(self, strategy: BackboneStrategy):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> BackboneStrategy:
        """
        The BackboneBlock maintains a reference to one of the Strategy objects. The
        BackboneBlock does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: BackboneStrategy):
        """
        To replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def build_model(self, input_shape):
        """Build a backbone for our net
        # Arguments:
            input_shape (list): Input image size and channels
            n_layers (int): Number of feature layers for SSD
            version (int): Supports ResNetv1 and v2
            n (int): Determines number of ResNet layers
                    (Default is ResNet50)
        # Returns
            model (Keras Model)
        """
        if type(self._strategy).__name__ == 'darknet53':
            return self._strategy.build(input_shape)
        elif type(self._strategy).__name__ == 'darknet19_tiny':
            return self._strategy.build(input_shape)

class darknet53(BackboneStrategy):
    def build(self, x):
        """ Build a darknet53 model
        Arguments: 
            x: an input tensor that can be an image
        """
        x = Conv2dBNLeakyReluLayer((3, 3,  3,  32))(x)
        x = Conv2dBNLeakyReluLayer((3, 3, 32,  64), down_sample=True)(x)

        for i in range(1):
            x = ResidualLayer(64,  32, 64)(x)

        x = Conv2dBNLeakyReluLayer((3, 3,  64, 128), down_sample=True)(x)

        for i in range(2):
            x = ResidualLayer(128,  64, 128)(x)

        x = Conv2dBNLeakyReluLayer((3, 3, 128, 256), down_sample=True)(x)

        for i in range(8):
            x = ResidualLayer(256, 128, 256)(x)

        route_1 = x
        x = Conv2dBNLeakyReluLayer((3, 3, 256, 512), down_sample=True)(x)

        for i in range(8):
            x = ResidualLayer(512, 256, 512)(x)

        route_2 = x
        x = Conv2dBNLeakyReluLayer((3, 3, 512, 1024), down_sample=True)(x)

        for i in range(4):
            x = ResidualLayer(1024, 512, 1024)(x)

        return route_1, route_2, x

class darknet19_tiny(BackboneStrategy):
    def build(self, x):
        """ Build a darknet19 tiny model
        Arguments: 
            x: an input tensor that can be an image
        """
        filters_shapes = [(3, 3, 3, 16), (3, 3, 16, 32), (3, 3, 32, 64), (3, 3, 64, 128), (3, 3, 128, 256)]
        for i, filter_shape in enumerate(filters_shapes):
            x = Conv2dBNLeakyReluLayer(filter_shape)(x)
            if i < 4:
                x = MaxPool2D(2, 2, 'same')(x)
        
        route_1 = x

        x = MaxPool2D(2, 2, 'same')(x)
        x = Conv2dBNLeakyReluLayer((3, 3, 256, 512))(x)
        x = MaxPool2D(2, 1, 'same')(x)
        x = Conv2dBNLeakyReluLayer((3, 3, 512, 1024))(x)

        return route_1, x
        