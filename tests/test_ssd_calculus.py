import unittest
import numpy as np

from compute.ssd_calculus import SSDCalculus

class TestSSDCalculus(unittest.TestCase):
    def setUp(self):
        self.compute = SSDCalculus()
    
    def test_anchor_sizes_returned_type_is_list(self):

        self.assertEqual(type(self.compute.anchor_sizes()), list)

    def test_anchor_sizes_dimensions(self):
        # test case n_layers = 10
        anchors = self.compute.anchor_sizes(10)
        anchors = np.array(anchors)
        self.assertEqual(anchors.shape, (10, 2))
        # test case n_layers = 4
        anchors = self.compute.anchor_sizes()
        anchors = np.array(anchors)
        self.assertEqual(anchors.shape, (4, 2))
    
    def test_centroid2minmax_and_minmax2centroid(self):
        # create random boxes (cx, cy, w, h)
        scaling_factor = 30
        np.random.seed(2021)
        boxes = np.random.rand(2, 2, 1, 4)*scaling_factor
        boxes = boxes.astype(int)
        # convert boxes from (cx, cy, w, h) format to (xmin, xmax, ymin, ymax) format and reverse
        minmax = self.compute.centroid2minmax(boxes)
        expected_minmax =   np.array([[16, 20, 17.5, 26.5], 
                            [26.5, 31.5, -8, 14], 
                            [18, 20, 22.5, 23.5], 
                            [27, 29, 10, 26]])
        # Reshape values equal to input array
        expected_minmax = expected_minmax.reshape(boxes.shape)
        self.assertEqual(np.alltrue(expected_minmax == minmax), True)
        centroid = self.compute.minmax2centroid(minmax)
        self.assertEqual(np.alltrue(boxes == centroid), True)

if __name__ == '__main__':
    unittest.main()
