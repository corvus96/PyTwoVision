import unittest

from pytwovision.recognition.selector import Recognizer
from pytwovision.recognition.yolov3_detector import ObjectDetectorYoloV3

class TestSelector(unittest.TestCase):

    def test_pattern_implementation(self):
        implementation = ObjectDetectorYoloV3('test_yolov3', 20)
        recognizer = Recognizer(implementation)
        self.assertEqual("test_yolov3", recognizer.implementation.model._name)
        
if __name__ == '__main__':
    unittest.main()
