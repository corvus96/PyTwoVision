import unittest

from pytwovision.recognition.selector import Recognizer
from pytwovision.recognition.yolov3_detector import ObjectDetectorYoloV3

class TestSelector(unittest.TestCase):
    def setUp(self):
        pass

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
