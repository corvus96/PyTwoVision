import unittest
import os
import glob
import cv2 as cv
import math as mt

from matplotlib import pyplot as plt

from pytwovision.input_output.camera import Camera
from pytwovision.stereo.standard_stereo import StandardStereo
from pytwovision.stereo.standard_stereo import StandardStereoBuilder
from pytwovision.image_process.frame_decorator import Frame
from pytwovision.image_process.resize import Resize
from pytwovision.stereo.match_method import Matcher
from pytwovision.stereo.match_method import StereoSGBM

class TestStandardStereo(unittest.TestCase):
    def setUp(self):
        self.left_camera = Camera("left_camera", "imx_219_A")
        self.right_camera = Camera("right_camera", "imx_219_B")
        self.stereo_pair_fisheye = StandardStereo(self.left_camera, self.right_camera)
        self.stereo_pair = StandardStereo(self.left_camera, self.right_camera, False)

    # def test_image_split(self):
    #     # adjust image path
    #     images_path, _ = os.path.splitext("tests/assets")
    #     # available formats
    #     formats = ['*.svg', '*.png', '*.jpg', '*.bmp', '*.jpeg', '*.raw']
    #     # get files names
    #     files_grabbed = [glob.glob((images_path + '/' + e)  if len(images_path) > 0 else e) for e in formats]
    #     # flatting files list
    #     images = [item for sublist in files_grabbed for item in sublist]

    #     for i, fname in enumerate(images):
    #         pair_img = cv.imread(fname,-1)
    #         height, width = pair_img.shape[:2]
    #         new_width = int(width/2)
    #         img_left = pair_img[0:height,0:new_width] #Y+H and X+W
    #         img_right = pair_img[0:height,new_width:width]
            
    #         cv.imwrite('tests/assets/left_camera_calibration/photo_{}.png'.format(i + 1), img_left)
    #         cv.imwrite('tests/assets/right_camera_calibration/photo_{}.png'.format(i + 1), img_right)

    # def test_video_split(self):
    #     vid = cv.VideoCapture("tests/assets/video/person_video.mp4")
    #     # by default VideoCapture returns float instead of int
    #     width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    #     half_width = int(width/2)
    #     height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    #     half_height = int(height/2)
    #     fps = int(vid.get(cv.CAP_PROP_FPS))
    #     codec = cv.VideoWriter_fourcc(*'XVID')
    #     out_left = cv.VideoWriter("tests/assets/video/left_person_video.mp4", codec, fps, (half_width, height))
    #     out_right = cv.VideoWriter("tests/assets/video/right_person_video.mp4", codec, fps, (half_width, height))
    #     while True:
    #         _, img = vid.read()
    #         if img is None:
    #             break
    #         left_img = img[0:height,0:half_width]
    #         right_img = img[0:height,half_width:width]
    #         out_left.write(left_img)
    #         out_right.write(right_img)

    # def test_resize(self):
    #     width = 640
    #     height = 720
    #     images_left, _ = os.path.splitext("tests/assets/left_camera_calibration")
    #     images_right, _ = os.path.splitext("tests/assets/right_camera_calibration")
    #     # available formats
    #     formats = ['*.svg', '*.png', '*.jpg', '*.bmp', '*.jpeg', '*.raw']
    #     # get files names
    #     filesL = [glob.glob((images_left + '/' + e)  if len(images_left) > 0 else e) for e in formats]
    #     filesR = [glob.glob((images_right + '/' + e)  if len(images_right) > 0 else e) for e in formats]
    #     # flatting files list
    #     imagesL = [item for sublist in filesL for item in sublist]
    #     imagesR = [item for sublist in filesR for item in sublist]
    #     imagesL = sorted(imagesL)
    #     imagesR = sorted(imagesR)
    #     for imgLeft, imgRight in zip(imagesL, imagesR):
    #         imgL = cv.imread(imgLeft)
    #         imgR = cv.imread(imgRight)
    #         frameL = Frame(imgL)
    #         frameR = Frame(imgR)
    #         imgL = Resize(frameL).apply(width, height)
    #         imgR = Resize(frameR).apply(width, height)
    #         cv.imwrite("{}".format(imgLeft), imgL)
    #         cv.imwrite("{}".format(imgRight), imgR)

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
        left = cv.imread("tests/assets/left_camera_calibration/photo_1.png")
        right = cv.imread("tests/assets/right_camera_calibration/photo_1.png")
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