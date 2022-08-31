import unittest
import os
import shutil

from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat
from pytwovision.datasets_loader.yolov3_dataset_generator import YoloV3DatasetGenerator

class TestYoloV3DatasetGenerator(unittest.TestCase):
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

    def test_load_annotations(self):
        anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        classes_full_path = os.path.join(self.work_dir, "{}.txt".format(self.classes_file))
        test_set = YoloV3DatasetGenerator(anno_out_full_path, classes_full_path)
        self.assertIsInstance(test_set.annotations, list) 
    
    def test_dataset_iter(self):
        anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        classes_full_path = os.path.join(self.work_dir, "{}.txt".format(self.classes_file))
        test_set = YoloV3DatasetGenerator(anno_out_full_path, classes_full_path, batch_size=1)
        
        count = 0
        for element in test_set:
            count += 1

        self.assertEqual(count, 5)
    
    def test_parse_annotation(self):
        anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        classes_full_path = os.path.join(self.work_dir, "{}.txt".format(self.classes_file))
        test_set = YoloV3DatasetGenerator(anno_out_full_path, classes_full_path, batch_size=1)
        for element in test_set.annotations:    
            image, bboxes = test_set.parse_annotation(element)
            self.assertEqual(image.shape, (416, 416, 3))
            self.assertEqual(bboxes.shape, (bboxes.shape[0], 5))
    
    def tearDown(self):
        try:
            shutil.rmtree(self.work_dir)
        except:
            pass
        
if __name__ == '__main__':
    unittest.main()
