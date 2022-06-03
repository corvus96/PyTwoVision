import unittest
import numpy as np

from pytwovision.models.layers.upsample_layer import UpsampleLayer

class TestUpsampleLayer(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.test_array = np.random.randint(0, 255, size=(3, 4, 4, 1))
    
    def test_output_shape(self):
        output = UpsampleLayer()(self.test_array)
        self.assertEqual((self.test_array.shape[0], self.test_array.shape[1] * 2, self.test_array.shape[2] * 2, self.test_array.shape[3]), output.shape)

if __name__ == '__main__':
    unittest.main()
