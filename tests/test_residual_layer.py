import unittest
import numpy as np

from pytwovision.models.layers.residual_layer import residual_layer

class TestResidualLayer(unittest.TestCase):
        def test_layer_output(self):
            output_depth = 48
            np.random.seed(2021)
            test_array = np.random.rand(1, 4, 4, 1)
            output = residual_layer(test_array, 32, 24, output_depth)

            self.assertEqual(output.shape, (1, 4, 4, output_depth))

            

if __name__ == '__main__':
    unittest.main()
