import unittest
import numpy as np
import tensorflow as tf

from pytwovision.models.layers.residual_layer import ResidualLayer

class TestResidualLayer(unittest.TestCase):
        def test_layer_output(self):
            output_depth = 48
            layer = ResidualLayer('test', 32, 24, output_depth, '1')
            np.random.seed(2021)
            test_array = np.random.rand(1, 4, 4, 1)
            output = layer(test_array)

            self.assertEqual(output.shape, (1, 4, 4, output_depth))

            

if __name__ == '__main__':
    unittest.main()
