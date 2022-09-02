
import unittest
import numpy as np
import tensorflow as tf

from py2vision.compute.yolov3_calculus import YoloV3Calculus
from py2vision.models.models_manager import ModelManager
from py2vision.models.blocks.backbone_block import BackboneBlock
from py2vision.models.blocks.backbone_block import darknet53

class TestYoloV3Calculus(unittest.TestCase):
    def setUp(self):
        self.compute = YoloV3Calculus()
        self.input_data = np.random.randint(0, 255, size=(416, 416, 3))
        self.box1 = np.array([[15, 66, 122, 70]])
        self.box2 = np.array([[20, 60, 22, 78]])

    def test_decode_minimal_output_size(self):
        training = True
        backbone_net = BackboneBlock(darknet53())
        model_manager = ModelManager()
        conv_tensors = model_manager.build_yolov3(backbone_net, 20)(self.input_data.shape)
        input_layer = conv_tensors[-1]
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors[:-1]):
            pred_tensor = self.compute.decode(conv_tensor, 20, i)
        
        self.assertEqual((pred_tensor.shape[0], pred_tensor.shape[1], pred_tensor.shape[2], pred_tensor.shape[3], pred_tensor.shape[4]), (None, 13, 13, 3, 25))
    
    def test_centroid2minmax_and_minmax2centroid(self):
        # create random boxes (cx, cy, w, h)
        scaling_factor = 30
        np.random.seed(2021)
        boxes = np.random.rand(2, 2, 1, 4)*scaling_factor
        boxes = boxes.astype(int)
        # convert boxes from (cx, cy, w, h) format to (xmin, ymin, xmax, ymax) format and reverse
        minmax = self.compute.centroid2minmax(boxes)
        expected_minmax =   np.array([[16, 17.5, 20, 26.5], 
                            [26.5, -8, 31.5, 14], 
                            [18, 22.5, 20, 23.5], 
                            [27, 10, 29, 26]])
        # Reshape values equal to input array
        expected_minmax = expected_minmax.reshape(boxes.shape)
        self.assertEqual(np.alltrue(expected_minmax == minmax), True)
        centroid = self.compute.minmax2centroid(minmax)
        self.assertEqual(np.alltrue(boxes == centroid), True)
    
    def test_best_bbox_iou(self):
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3

        box1_area = (self.box1[0, xmax] - self.box1[0, xmin]) * (self.box1[0, ymax] - self.box1[0, ymin])
        box2_area = (self.box2[0, xmax] - self.box2[0, xmin]) * (self.box2[0, ymax] - self.box2[0, ymin])

        left_up = np.maximum(self.box1[0, :2], self.box2[0, :2])
        right_down = np.minimum(self.box1[0, 2:], self.box2[0, 2:])
        
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = box1_area + box2_area - inter_area

        iou = 1.0 * inter_area / union_area

        self.assertEqual(self.compute.best_bboxes_iou(self.box1, self.box2),  tf.constant([iou]))
    
    def test_giou_vs_ciou_vs_iou(self):
        # scaling_factor = 30
        # np.random.seed(2021)
        # box1 = np.random.rand(1,4)*scaling_factor
        # np.random.seed(1)
        # box2 = np.random.rand(1,4)*scaling_factor
        giou = self.compute.bbox_giou(self.box1, self.box2)
        ciou = self.compute.bbox_ciou(self.box1, self.box2)
        iou = self.compute.bbox_iou(self.box1, self.box2)
        best_iou = self.compute.best_bboxes_iou(self.box1, self.box2)
        
        #test giou vs ciou
        self.assertGreater(tf.abs(best_iou - ciou), tf.abs(best_iou - giou))
        #test giou vs iou
        self.assertGreater(tf.abs(best_iou - iou), tf.abs(best_iou - giou))
        #test ciou vs iou
        self.assertGreater(tf.abs(best_iou - iou), tf.abs(best_iou - ciou))
        
if __name__ == '__main__':
    unittest.main()
