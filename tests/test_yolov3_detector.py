
from typing import List
import unittest
import numpy as np
import tensorflow as tf
import os
import shutil

from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat
from pytwovision.datasets_loader.yolov3_dataset_generator import YoloV3DatasetGenerator
from pytwovision.recognition.yolov3_detector import ObjectDetectorYoloV3

class TestObjectDetectorYoloV3(unittest.TestCase):
    def setUp(self):
        self.anno_out_file = "annotations_formated"
        self.xml_path = "tests/test_dataset/annotations"
        self.classes_file = "test_dataset_generator"
        self.work_dir = "tests/test_dataset/to_generator_test"
        self.images_path = "tests/test_dataset/images"

        try:
            os.mkdir(self.work_dir)
        except:
            pass

        #create annotations formated
        parser = XmlParser()
        anno_format = YoloV3AnnotationsFormat()
        parser.parse(anno_format, self.xml_path, self.anno_out_file, self.classes_file, self.images_path, self.work_dir)
        self.anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        self.classes_full_path = os.path.join(self.work_dir, "{}.txt".format(self.classes_file))

    def test_model_created(self):
        yolov3 = ObjectDetectorYoloV3("test_yolov3", 4)
        yolov3_tiny = ObjectDetectorYoloV3("test_yolov3_tiny", 4, version="yolov3_tiny")
        self.assertEqual(yolov3.model._name, "test_yolov3")
        self.assertEqual(yolov3_tiny.model._name, "test_yolov3_tiny")
    
    def test_train(self):
        pass
    
    def tearDown(self):
        try:
            shutil.rmtree(self.work_dir)
        except:
            pass
        
if __name__ == '__main__':
    unittest.main()