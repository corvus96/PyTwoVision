import unittest
import numpy as np
import tensorflow as tf

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import Conv2dBNLeakyReluLayer

class TestConv2dBNLeakyReluLayer(unittest.TestCase):
        def test_add_downsample_layer(self):
            layer = Conv2dBNLeakyReluLayer((1, 1, 58, 40), down_sample=True)
            self.assertEqual(type(layer), Conv2dBNLeakyReluLayer)
        
        def test_output_shape(self):
            test_array = np.random.rand(3, 4, 4, 1) * 5
            layer = Conv2dBNLeakyReluLayer((1, 1, 58, 40), down_sample=True)(test_array)
            self.assertEqual(layer.shape, (3, 3, 3, 40))


if __name__ == '__main__':
    unittest.main()
