import unittest
import os
import shutil

from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat
from pytwovision.utils.annotations_helper import AnnotationsHelper

class TestAnnotationsParser(unittest.TestCase):
    def setUp(self) -> None:
        
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
        
    def test_split(self):
        training_percentages = [0.1, 0.4, 0.5, 0.8, 1]
        anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        anno_helper = AnnotationsHelper(anno_out_full_path)

        for training_pe in training_percentages:
            train, test = anno_helper.split(training_pe)
            anno_helper.export(train, os.path.join(self.work_dir, "train.txt"))
            anno_helper.export(test, os.path.join(self.work_dir, "test.txt"))
            self.assertEqual(len(train), int(len(anno_helper.annotations)*training_pe))

            
    def test_shuffle(self):
        anno_out_full_path = os.path.join(self.work_dir, "{}.txt".format(self.anno_out_file))
        anno_helper = AnnotationsHelper(anno_out_full_path)
        anno_helper.shuffle()
        self.assertEqual(len(anno_helper.annotations), 5)

    def tearDown(self):
        try:
            shutil.rmtree(self.work_dir)
        except:
            pass



if __name__ == '__main__':
    unittest.main()
