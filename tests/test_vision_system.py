import unittest
import os

from pytwovision.input_output.vision_system import VisionSystem
from pytwovision.input_output.camera import Camera
from pytwovision.stereo.standard_stereo import StandardStereo
from pytwovision.stereo.match_method import Matcher, StereoSGBM
from pytwovision.recognition.yolov3_detector import ObjectDetectorYoloV3
from pytwovision.recognition.selector import Recognizer

from matplotlib import pyplot as plt

class TestVisionSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.anno_out_file = "annotations_formated"
        self.xml_path = "tests/test_dataset/annotations"
        self.classes_file = "tests/test_dataset/classes/coco.names"
        self.work_dir = "tests/test_dataset/to_generator_test"
        self.images_path = "tests/test_dataset/images"
        stereo_maps_path = "stereoMap"
        self.stereo_maps_path = stereo_maps_path + ".xml"
        try:
            os.mkdir(self.work_dir)
        except:
            pass
        left_camera = Camera("left_camera", "tests/assets/photo/left/left_indoor_photo_5.png")
        right_camera = Camera("right_camera", "tests/assets/photo/right/right_indoor_photo_5.png")
        self.stereo_pair_fisheye = StandardStereo(left_camera, right_camera)
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True, export_file_name=stereo_maps_path)
        self.stereo_pair_fisheye.Q[0, 3] = -320
        self.stereo_pair_fisheye.Q[1, 3] = -360
        self.stereo_pair_fisheye.Q[3, 2] = -1 / 65e-2
        self.stereo_pair_fisheye.Q[2, 3] = 309
        self.stereo_pair_fisheye.Q[3, 3] = 0

    def test_positioning_on_video_flow(self):
        sgbm = StereoSGBM(min_disp=-32, max_disp=32, window_size=3, p1=89, p2=487, pre_filter_cap=14, speckle_window_size=114, speckle_range=1, uniqueness_ratio=3, disp_12_max_diff=-38)
        lmbda = 8214
        sigma = 1.5890
        matcher = Matcher(sgbm)
        left_camera = Camera("left_camera", "tests/assets/video/left_indoor_video.mp4")
        right_camera = Camera("right_camera", "tests/assets/video/right_indoor_video.mp4")
        vis_sys = VisionSystem(left_camera, right_camera, self.stereo_maps_path, matcher, self.stereo_pair_fisheye.Q)
        yolov3 = ObjectDetectorYoloV3("test", 80, training=False)
        recognizer = Recognizer(yolov3)
        #link_yolov3_weights = "https://pjreddie.com/media/files/yolov3.weights"
        #wget.download(link_yolov3_weights)
        #weights_file = os.path.basename(link_yolov3_weights)
        recognizer.restore_weights("yolov3.weights")
        model = recognizer.get_model()
        vis_sys.realtime_or_video_pipeline(model, self.classes_file, os.path.join(self.work_dir, "test_indoor_position.mp4"),  lmbda=lmbda, sigma=sigma, downsample_for_match=None, show_window=True, score_threshold=0.5, iou_threshold=0.6, otsu_thresh_inverse=True)
        #os.remove(weights_file)
        
    def test_positioning_on_image_flow(self):
        # A
        # sgbm = StereoSGBM(min_disp=-32, max_disp=32, window_size=3, p1=89, p2=487, pre_filter_cap=14, speckle_window_size=114, speckle_range=1, uniqueness_ratio=3, disp_12_max_diff=-38)
        # lmbda = 8214
        # sigma = 1.5890
        # B
        # sgbm = StereoSGBM(min_disp=-33, max_disp=32, window_size=3, p1=125, p2=932, pre_filter_cap=57, speckle_window_size=120, speckle_range=9, uniqueness_ratio=3, disp_12_max_diff=-38)
        # matcher = Matcher(sgbm)
        # lmbda = 19132
        # sigma = 1.046
        # m
        sgbm = StereoSGBM(min_disp=-32, max_disp=32, window_size=3, p1=107, p2=710, pre_filter_cap=36, speckle_window_size=117, speckle_range=5, uniqueness_ratio=3, disp_12_max_diff=-38)
        matcher = Matcher(sgbm)
        lmbda = 13673
        sigma = 1.3175
        matcher = Matcher(sgbm)
        left_camera = Camera("left_camera", "tests/assets/photo/left/left_plant_1.png")
        right_camera = Camera("right_camera", "tests/assets/photo/right/right_plant_1.png")
        vis_sys = VisionSystem(left_camera, right_camera, self.stereo_maps_path, matcher, self.stereo_pair_fisheye.Q)
        yolov3 = ObjectDetectorYoloV3("test", 80, training=False)
        recognizer = Recognizer(yolov3)
        #link_yolov3_weights = "https://pjreddie.com/media/files/yolov3.weights"
        #wget.download(link_yolov3_weights)
        #weights_file = os.path.basename(link_yolov3_weights)
        recognizer.restore_weights("yolov3.weights")
        model = recognizer.get_model()
        vis_sys.image_pipeline(model, self.classes_file, os.path.join(self.work_dir, "test_position.jpg"),  lmbda=lmbda, sigma=sigma, downsample_for_match=None, show_window=True, score_threshold=0.5, iou_threshold=0.6, otsu_thresh_inverse=True, text_colors=(0, 0, 0))
        #os.remove(weights_file)

    def tearDown(self):
        # try:
        #     os.remove(self.stereo_maps_path)
        #     shutil.rmtree(self.work_dir)
        #     tf.keras.backend.clear_session()
        # except:
        #     pass
        pass