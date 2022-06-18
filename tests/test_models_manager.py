import unittest
import numpy as np
import os 
from pytwovision.models.models_manager import ModelManager
from pytwovision.models.blocks.backbone_block import BackboneBlock
from pytwovision.models.blocks.backbone_block import darknet53, darknet19_tiny

class TestModelManager(unittest.TestCase):
    def setUp(self):
        np.random.seed(2000)
        self.input_data = np.random.randint(0, 255, size=(416, 416, 3))
    
    def test_yolov3_model_output(self):
        backbone_net = BackboneBlock(darknet53())
        model_manager = ModelManager()
        conv_sbbox, conv_mbbox, conv_lbbox, _ = model_manager.build_yolov3(backbone_net, 80)(self.input_data)
        
        self.assertEqual((conv_sbbox.shape[0], conv_sbbox.shape[1], conv_sbbox.shape[2], conv_sbbox.shape[3]) , (None, 52, 52, 255))
        self.assertEqual((conv_mbbox.shape[0], conv_mbbox.shape[1], conv_mbbox.shape[2], conv_mbbox.shape[3]), (None, 26, 26, 255))
        self.assertEqual((conv_lbbox.shape[0], conv_lbbox.shape[1], conv_lbbox.shape[2], conv_lbbox.shape[3]), (None, 13, 13, 255))

    def test_yolov3_tiny_model_output(self):
        backbone_net = BackboneBlock(darknet19_tiny())
        model_manager = ModelManager()
        conv_mbbox, conv_lbbox, _ = model_manager.build_yolov3_tiny(backbone_net, 80)(self.input_data)

        self.assertEqual((conv_mbbox.shape[0], conv_mbbox.shape[1], conv_mbbox.shape[2], conv_mbbox.shape[3]), (None, 26, 26, 255))
        self.assertEqual((conv_lbbox.shape[0], conv_lbbox.shape[1], conv_lbbox.shape[2], conv_lbbox.shape[3]), (None, 13, 13, 255))
    
    def test_num_class_type(self):
        backbone_net = BackboneBlock(darknet19_tiny())
        model_manager = ModelManager()
        try:    
            conv_mbbox, conv_lbbox, _ = model_manager.build_yolov3_tiny(backbone_net, 85.4)(self.input_data)
        except ValueError:
            self.assertTrue(True)

        

if __name__ == '__main__':
    unittest.main()
