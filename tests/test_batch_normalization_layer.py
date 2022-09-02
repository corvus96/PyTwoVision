import unittest
import numpy as np
import tensorflow as tf

from py2vision.models.layers.batch_normalization_layer import BatchNormalization

class TestBatchNormalizationLayer(unittest.TestCase):
        def test_std_desviation_decrease(self):
            std_output_less_than_input = []
            for i in range(100):
                test_array = np.random.rand(3, 4, 4) * 5
                output = BatchNormalization()(test_array)
                std_output_less_than_input.append(tf.cast(tf.math.reduce_std(output), tf.float32) < tf.cast(tf.math.reduce_std(test_array), tf.float32))
                self.assertListEqual(std_output_less_than_input, [True] * len(std_output_less_than_input))

        def test_mean_decrease(self):
            mean_output_less_than_input = []
            for i in range(100):
                test_array = np.random.rand(3, 4, 4) * 5
                output = BatchNormalization()(test_array)
                mean_output_less_than_input.append(tf.cast(tf.reduce_mean(output), tf.float32) < tf.cast(tf.reduce_mean(test_array), tf.float32))
                self.assertListEqual(mean_output_less_than_input, [True] * len(mean_output_less_than_input))

if __name__ == '__main__':
    unittest.main()
