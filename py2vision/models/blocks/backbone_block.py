from abc import ABC, abstractmethod

from tensorflow.keras.layers import MaxPool2D

from py2vision.models.layers.conv2d_bn_leaky_relu_layer import conv2d_bn_leaky_relu_layer
from py2vision.models.layers.residual_layer import residual_layer

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
        """Build a backbone for our net.
        
        Args:
            input_shape: Input image size and channels.

        Returns
            model (Keras Model)
        """
        if type(self._strategy).__name__ == 'darknet53':
            return self._strategy.build(input_shape)
        elif type(self._strategy).__name__ == 'darknet19_tiny':
            return self._strategy.build(input_shape)

class darknet53(BackboneStrategy):
    def build(self, x):
        """ Build a darknet53 model.

        Args: 
            x: an input tensor that can be an image.
            
        Returns:
            three branches of darknet53.
        """
        x = conv2d_bn_leaky_relu_layer(x, (3, 3,  3,  32))
        x = conv2d_bn_leaky_relu_layer(x, (3, 3, 32,  64), downsample=True)

        for i in range(1):
            x = residual_layer(x, 64,  32, 64)

        x = conv2d_bn_leaky_relu_layer(x, (3, 3,  64, 128), downsample=True)

        for i in range(2):
            x = residual_layer(x, 128,  64, 128)

        x = conv2d_bn_leaky_relu_layer(x, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            x = residual_layer(x, 256, 128, 256)

        route_1 = x
        x = conv2d_bn_leaky_relu_layer(x, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            x = residual_layer(x, 512, 256, 512)

        route_2 = x
        x = conv2d_bn_leaky_relu_layer(x, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            x = residual_layer(x, 1024, 512, 1024)

        return route_1, route_2, x

class darknet19_tiny(BackboneStrategy):
    def build(self, x):
        """ Build a darknet19 tiny model.
        
        Args: 
            x: an input tensor that can be an image.
        
        Returns: 
            Two branches of darknet19 tiny.
        """
        filters_shapes = [(3, 3, 3, 16), (3, 3, 16, 32), (3, 3, 32, 64), (3, 3, 64, 128), (3, 3, 128, 256)]
        for i, filter_shape in enumerate(filters_shapes):
            x = conv2d_bn_leaky_relu_layer(x, filter_shape)
            if i < 4:
                x = MaxPool2D(2, 2, 'same')(x)
        
        route_1 = x

        x = MaxPool2D(2, 2, 'same')(x)
        x = conv2d_bn_leaky_relu_layer(x, (3, 3, 256, 512))
        x = MaxPool2D(2, 1, 'same')(x)
        x = conv2d_bn_leaky_relu_layer(x, (3, 3, 512, 1024))

        return route_1, x
        