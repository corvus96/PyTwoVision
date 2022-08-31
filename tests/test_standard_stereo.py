import unittest
import os
import cv2 as cv
import math as mt

from pytwovision.input_output.camera import Camera
from pytwovision.stereo.standard_stereo import StandardStereo
from pytwovision.stereo.standard_stereo import StandardStereoBuilder
from pytwovision.stereo.match_method import Matcher
from pytwovision.stereo.match_method import StereoSGBM

class TestStandardStereo(unittest.TestCase):
    def setUp(self):
        self.left_camera = Camera("left_camera", "imx_219_A")
        self.right_camera = Camera("right_camera", "imx_219_B")
        self.stereo_pair_fisheye = StandardStereo(self.left_camera, self.right_camera)
        self.stereo_pair = StandardStereo(self.left_camera, self.right_camera, False)

    def test_calibrate_without_calibrate_individual_cameras_parameters(self):
        is_ok = True
        try:
            self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        except:
            is_ok = False
        self.assertTrue(is_ok)

    def test_fish_eye_type(self):
        is_not_boolean = False
        try:
            StandardStereo(self.left_camera, self.right_camera, None)
        except:
            is_not_boolean = True
        self.assertTrue(is_not_boolean)
    
    def test_rectification_flow(self):
        is_ok = True
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        try:
            self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=False)
        except:
            is_ok = False
        self.assertTrue(is_ok)
    
    def test_get_stereo_maps(self):
        is_ok = True
        output_path = "stereoMap.xml"
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True)
        try:
            self.stereo_pair_fisheye.get_stereo_maps(output_path)
        except:
            is_ok = False
        os.remove(output_path)
        self.assertTrue(is_ok)

    def test_preprocess_with_fisheye_flow(self):
        output_path = "stereoMap.xml"
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True)
        builder = StandardStereoBuilder(self.left_camera, self.right_camera, output_path)
        left = cv.imread("tests/assets/photo/left/left_indoor_photo_3.png")
        right = cv.imread("tests/assets/photo/right/right_indoor_photo_3.png")
        is_ok = True
        try:
            rect_left, rect_right = builder.pre_process(left, right, downsample=None)
        except:
            is_ok = False
        os.remove(output_path)
        self.assertTrue(is_ok)

    def test_preprocess_output_size(self):
        output_path = "stereoMap.xml"
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True)
        builder = StandardStereoBuilder(self.left_camera, self.right_camera, output_path)
        left = cv.imread("tests/assets/left_camera_calibration/photo_1.png")
        right = cv.imread("tests/assets/right_camera_calibration/photo_1.png")
        for down_factor in [2**p for p in range(1, 7)]:
            rect_left, _ = builder.pre_process(left, right, downsample=down_factor)
            self.assertEqual(rect_left.shape, (mt.ceil(left.shape[0]/ down_factor), mt.ceil(left.shape[1]/ down_factor)))

        os.remove(output_path)
    
    def test_epilines(self):
        output_path = "stereoMap.xml"
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True)
        builder = StandardStereoBuilder(self.left_camera, self.right_camera, output_path)
        left = cv.imread("tests/assets/left_camera_calibration/photo_1.png")
        right = cv.imread("tests/assets/right_camera_calibration/photo_1.png")
        rect_left, rect_right = builder.pre_process(left, right, downsample=1)
        is_ok = True
        try: 
            left, right = builder.find_epilines(rect_left, rect_right)
        except:
            is_ok = False
        os.remove(output_path)
        self.assertTrue(is_ok)
    
    def test_matching(self):
        output_path = "stereoMap.xml"
        self.stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
        self.stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True)
        builder = StandardStereoBuilder(self.left_camera, self.right_camera, output_path)
        left = cv.imread("tests/assets/photo/left/left_indoor_photo_2.png")
        right = cv.imread("tests/assets/photo/right/right_indoor_photo_2.png")
        rect_left, rect_right = builder.pre_process(left, right, downsample=1)
        matcher = Matcher(StereoSGBM())
        is_ok = True
        try:
            builder.match(rect_left, rect_right, matcher)
        except:
            is_ok = False
        os.remove(output_path)
        self.assertTrue(is_ok)
        
if __name__ == '__main__':
    unittest.main()