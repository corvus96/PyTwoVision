import unittest
import numpy as np
import tensorflow as tf
import os
import shutil
import wget
import glob

from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat
from pytwovision.recognition.yolov3_detector import ObjectDetectorYoloV3
from pytwovision.utils.annotations_helper import AnnotationsHelper
from pytwovision.utils.label_utils import read_class_names
from pytwovision.datasets_loader.yolov3_dataset_generator import YoloV3DatasetGenerator

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
    
    def test_train_yolov3(self):
        yolov3 = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_full_path)), training=True)
        anno_helper = AnnotationsHelper(self.anno_out_full_path)
        train_set, test_set = anno_helper.split(0.5)
        train_dataset_path = os.path.join(self.work_dir, "train.txt")
        test_dataset_path = os.path.join(self.work_dir, "test.txt")
        anno_helper.export(train_set, train_dataset_path)
        anno_helper.export(test_set, test_dataset_path)
        could_train = True
        try:
            yolov3.train(train_dataset_path, test_dataset_path, self.classes_full_path, epochs=1, batch_size=1, checkpoint_path=os.path.join(self.work_dir, "checkpoints"), log_dir=os.path.join(self.work_dir, "logs"))
        except:
            could_train = False
        self.assertTrue(could_train)

    def test_train_yolov3_tiny(self):
        yolov3_tiny = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_full_path)), training=True)
        anno_helper = AnnotationsHelper(self.anno_out_full_path)
        train_set, test_set =anno_helper.split(0.5)
        train_dataset_path = os.path.join(self.work_dir, "train.txt")
        test_dataset_path = os.path.join(self.work_dir, "test.txt")
        anno_helper.export(train_set, train_dataset_path)
        anno_helper.export(test_set, test_dataset_path)
        could_train = True
        try:
            yolov3_tiny.train(train_dataset_path, test_dataset_path, self.classes_full_path, epochs=1, batch_size=1, checkpoint_path=os.path.join(self.work_dir, "checkpoints"), log_dir=os.path.join(self.work_dir, "logs"))
        except:
            could_train = False
        self.assertTrue(could_train)

    def test_inference_yolov3(self):
        # yolov3_tiny = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_full_path)), training=True, version="yolov3_tiny")
        link_yolov3_weights = "https://pjreddie.com/media/files/yolov3.weights"
        wget.download(link_yolov3_weights)
        weights_file = os.path.basename(link_yolov3_weights)
        yolov3 = ObjectDetectorYoloV3("test", 80)
        yolov3.restore_weights(weights_file)

        image = np.random.choice(glob.glob(self.images_path +'/*.jpg'))
        os.remove(weights_file)
        bboxes = yolov3.inference(image)
        self.assertEqual(np.asarray(bboxes).shape, (np.asarray(bboxes).shape[0], 6))
    
    def test_inference_yolov3_tiny(self):
        # yolov3_tiny = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_full_path)), training=True, version="yolov3_tiny")
        link_yolov3_tiny_weights = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        wget.download(link_yolov3_tiny_weights)
        weights_file = os.path.basename(link_yolov3_tiny_weights)
        yolov3 = ObjectDetectorYoloV3("test", 80, version="yolov3_tiny")
        yolov3.restore_weights(weights_file)

        image = np.random.choice(glob.glob(self.images_path +'/*.jpg'))
        os.remove(weights_file)
        bboxes = yolov3.inference(image)
        self.assertEqual(np.asarray(bboxes).shape, (np.asarray(bboxes).shape[0], 6))

    def test_restore_darknet53_weights(self):
        link_yolov3_weights = "https://pjreddie.com/media/files/yolov3.weights"
        wget.download(link_yolov3_weights)
        weights_file = os.path.basename(link_yolov3_weights)
        yolov3 = ObjectDetectorYoloV3("test", 80)
        restore = True
        try:
            yolov3.restore_weights(weights_file)
        except:
            restore = False

        self.assertTrue(restore)
        os.remove(weights_file)
    
    def test_restore_darknet19_tiny_weights(self):
        link_yolov3_tiny_weights = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        wget.download(link_yolov3_tiny_weights)
        weights_file = os.path.basename(link_yolov3_tiny_weights)
        yolov3_tiny = ObjectDetectorYoloV3("test", 80, version="yolov3_tiny")
        restore = True
        try:
            yolov3_tiny.restore_weights(weights_file)
        except:
            restore = False

        self.assertTrue(restore)
        os.remove(weights_file)
    
    def test_print_summary(self):
        yolov3 = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_full_path)))
        printed = True
        try:
            yolov3.print_summary()
        except:
            printed = False
        self.assertTrue(printed)


    def test_evaluate(self):
        yolov3 = ObjectDetectorYoloV3("test", len(read_class_names(self.classes_full_path)))
        anno_helper = AnnotationsHelper(self.anno_out_full_path)
        _, test_set = anno_helper.split(0.2)
        test_dataset_path = os.path.join(self.work_dir, "test.txt")
        anno_helper.export(test_set, test_dataset_path)
        test_set = YoloV3DatasetGenerator(test_dataset_path, self.classes_full_path)
        evaluation = yolov3.evaluate(yolov3.model, test_set, self.classes_full_path)
        self.assertTrue(True if evaluation >= 0 and evaluation <= 100 else False)

    def tearDown(self):
        try:
            shutil.rmtree(self.work_dir)
            tf.keras.backend.clear_session()
            if os.path.exists("mAP/ground-truth"): shutil.rmtree("mAP")
            if os.path.exists("logs"): shutil.rmtree("logs")
        except:
            pass
        
if __name__ == '__main__':
    unittest.main()
