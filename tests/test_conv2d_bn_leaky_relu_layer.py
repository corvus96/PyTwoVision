import unittest
import numpy as np
import tensorflow as tf

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import Conv2dBNLeakyReluLayer

class TestConv2dBNLeakyReluLayer(unittest.TestCase):
        def test_postfix_type(self):
            postfix_types = [1, '1', 1.1]
            for postfix in postfix_types:
                layer = Conv2dBNLeakyReluLayer('test_type', (1, 1, 58, 40), postfix=postfix)
                self.assertEqual(type(layer), Conv2dBNLeakyReluLayer)
        
        def test_add_downsample_layer(self):
            layer = Conv2dBNLeakyReluLayer('test_bn',  (1, 1, 58, 40), down_sample=True, postfix='1')
            self.assertEqual(type(layer), Conv2dBNLeakyReluLayer)

if __name__ == '__main__':
    unittest.main()
