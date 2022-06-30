
from sys import implementation
from typing import List
import unittest
import numpy as np
import tensorflow as tf
import os
import shutil

from pytwovision.recognition.selector import Recognizer
from pytwovision.recognition.yolov3_detector import ObjectDetectorYoloV3

class TestSelector(unittest.TestCase):
    def setUp(self):
        pass
        # self.anno_out_file = "annotations_formated"
        # self.xml_path = "tests/test_dataset/annotations"
        # self.classes_file = "test_dataset_generator"
        # self.work_dir = "tests/test_dataset/to_generator_test"
        # self.images_path = "tests/test_dataset/images"

        # try:
        #     os.mkdir(self.work_dir)
        # except:
        #     pass

        # #create annotations formated
        # parser = XmlParser()
        # anno_format = YoloV3AnnotationsFormat()
        # parser.parse(anno_format, self.xml_path, self.anno_out_file, self.classes_file, self.images_path, self.work_dir)
        # self.anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        # self.classes_full_path = os.path.join(self.work_dir, "{}.txt".format(self.classes_file))

    def test_pattern_implementation(self):
        implementation = ObjectDetectorYoloV3('test_yolov3', 20)
        recognizer = Recognizer(implementation)
        self.assertEqual("test_yolov3", recognizer.implementation.model._name)
    
    def tearDown(self):
        pass
        # try:
        #     shutil.rmtree(self.work_dir)
        # except:
        #     pass
        
if __name__ == '__main__':
    unittest.main()
