import unittest
import numpy as np

from py2vision.models.layers.conv2d_bn_leaky_relu_layer import conv2d_bn_leaky_relu_layer

class TestConv2dBNLeakyReluLayer(unittest.TestCase):
        
    def test_output_shape(self):
        test_array = np.random.rand(3, 4, 4, 1) * 5
        layer = conv2d_bn_leaky_relu_layer(test_array, (1, 1, 58, 40), downsample=True)
        self.assertEqual(layer.shape, (3, 3, 3, 40))

if __name__ == '__main__':
    unittest.main()
