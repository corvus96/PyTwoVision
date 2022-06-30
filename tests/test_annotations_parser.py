import unittest
import os

from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat


class TestAnnotationsParser(unittest.TestCase):
    def setUp(self) -> None:
        self.anno_out_file = "test_anno_file"
        self.xml_path = "tests/test_dataset/annotations"
        self.classes_out_file = "test_classes"
        self.work_dir = "tests"
        self.images_path = "tests/test_dataset/images"
    def test_no_xml_annotations(self):
        parser = XmlParser()
        anno_format = YoloV3AnnotationsFormat()
        xml_path = "bad_path"
        no_file_check = False
        try:
            parser.parse(anno_format, xml_path, self.anno_out_file, self.classes_out_file, self.images_path, self.work_dir)
        except FileNotFoundError:
            no_file_check = True
        
        if no_file_check is False: self.assertTrue(False)
        else: self.assertTrue(True)
        
    def test_xml_parser_output_files(self):
        parser = XmlParser()
        anno_format = YoloV3AnnotationsFormat()
        parser.parse(anno_format, self.xml_path, self.anno_out_file, self.classes_out_file, self.images_path, self.work_dir)
        
        cls_file = os.path.exists(os.path.join(self.work_dir, self.classes_out_file + ".txt"))
        anno_file = os.path.exists(os.path.join(self.work_dir, self.anno_out_file + ".txt"))

        self.assertTrue(cls_file)
        self.assertTrue(anno_file)
    
    def tearDown(self):
        try:
            os.remove(os.path.join(self.work_dir, self.classes_out_file + ".txt"))
            os.remove(os.path.join(self.work_dir, self.anno_out_file + ".txt"))
        except:
            pass



if __name__ == '__main__':
    unittest.main()
