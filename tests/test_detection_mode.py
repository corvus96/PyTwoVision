import unittest
import os
import shutil
import tensorflow as tf

from py2vision.recognition.detection_mode import DetectImage, DetectRealTime, DetectRealTimeMP
from py2vision.recognition.selector import Recognizer
from py2vision.recognition.yolov3_detector import ObjectDetectorYoloV3
from py2vision.utils.label_utils import read_class_names
from py2vision.input_output.camera import Camera

class TestDetectionMode(unittest.TestCase):
    def setUp(self):
        self.images_path = "tests/test_dataset/images"
        self.classes_path = "tests/test_dataset/classes/coco.names"
        self.voc_classes_path = "tests/test_dataset/classes/VOC_classes.txt"
        self.work_dir = "tests/test_dataset/outputs"
        try:
            os.mkdir(self.work_dir)
        except:
            pass

    def test_detect_realtime_output_path_not_is_mp4(self):
        yolov3 = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_path)))
        recognizer = Recognizer(yolov3)
        model = recognizer.get_model()
        detector =  DetectRealTime()
        cam = Camera("main_camera_test", 0)
        not_is_mp4 = False
        try:
            detector.detect(model, cam, self.classes_path, os.path.join(self.work_dir, "video_test.avi"))
        except ValueError:
            not_is_mp4 = True
        self.assertTrue(not_is_mp4)
    
    def test_detect_realtime_mp_output_path_not_is_mp4(self):
        yolov3 = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_path)))
        recognizer = Recognizer(yolov3)
        model = recognizer.get_model()
        detector =  DetectRealTimeMP()
        cam = Camera("main_camera_test", 0)
        not_is_mp4 = False
        try:
            detector.detect(model, cam, self.classes_path, os.path.join(self.work_dir, ""))
        except ValueError:
            not_is_mp4 = True
        self.assertTrue(not_is_mp4)
    
    def test_detect_image(self):
        yolov3 = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_path)))
        recognizer = Recognizer(yolov3)
        model = recognizer.get_model()
        detector =  DetectImage()
        not_is_compatible = False
        try:
            detector.detect(model, "tests/test_dataset/images/2007_000027.jpg", self.classes_path, os.path.join(self.work_dir, "test_out.jpag"), show=False)
        except ValueError:
            not_is_compatible = True
        self.assertTrue(not_is_compatible)

    def tearDown(self):
        try:
            shutil.rmtree(self.work_dir)
            tf.keras.backend.clear_session()
        except:
            pass
        
if __name__ == '__main__':
    unittest.main()
