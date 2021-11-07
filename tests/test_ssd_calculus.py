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
    
    def test_not_intersection(self):
        scaling_factor = 30
        np.random.seed(2021)
        box1 = np.random.rand(1,4)*scaling_factor
        np.random.seed(2022)
        box2 = np.random.rand(1,4)*scaling_factor
        intersection_area = self.compute.intersection(box1, box2)
        self.assertEqual(intersection_area[0][0], 0)
    
    def test_intersection_area(self):
        box1 = np.array([[20, 60, 10, 70]])
        box2 = np.array([[10, 30, 40, 100]])
        intersection_area = self.compute.intersection(box1, box2)
        self.assertEqual(intersection_area[0][0], 300)
    
    def test_union_without_intersection(self):
        scaling_factor = 30
        np.random.seed(2021)
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3
        box1 = np.random.rand(1,4)*scaling_factor
        box1_area = (box1[:, xmax] - box1[:, xmin]) * (box1[:, ymax] - box1[:, ymin])
        np.random.seed(2010)
        box2 = np.random.rand(1,4)*scaling_factor
        box2_area = (box2[:, xmax] - box2[:, xmin]) * (box2[:, ymax] - box2[:, ymin])
        intersection_area = self.compute.intersection(box1, box2)
        union_area = self.compute.union(box1, box2, intersection_area)
        self.assertEqual(union_area, box1_area + box2_area)
    
    def test_union_with_intersection(self):
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3
        box1 = np.array([[20, 60, 10, 70]])
        box1_area = (box1[:, xmax] - box1[:, xmin]) * (box1[:, ymax] - box1[:, ymin])
        box2 = np.array([[10, 30, 40, 100]])
        box2_area = (box2[:, xmax] - box2[:, xmin]) * (box2[:, ymax] - box2[:, ymin])
        intersection_area = self.compute.intersection(box1, box2)
        union_area = self.compute.union(box1, box2, intersection_area)
        self.assertEqual(union_area, box1_area + box2_area - intersection_area)
    
    def test_iou_better_choice(self):
        box1 = np.array([[20, 60, 10, 70]])
        box2 = np.array([[10, 30, 40, 100]])
        iou_a = self.compute.iou(box1, box2)
        # Better choice
        iou_b = self.compute.iou(box1, box1)
        self.assertGreater(iou_b, iou_a)

        

if __name__ == '__main__':
    unittest.main()
