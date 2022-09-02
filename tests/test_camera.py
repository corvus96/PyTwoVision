import unittest
import os

from py2vision.input_output.camera import Camera

class TestCamera(unittest.TestCase):
    def setUp(self):
        self.fisheye_camera = Camera("fisheye", "IMX219")
        self.normal_camera = Camera("normal", "IMX219")

    def test_calibrate_fisheye(self):
        is_ok = True
        try:
            self.fisheye_camera.calibrate("tests/assets/right_camera_calibration", show=False, export_file=False)
        except:
            is_ok = False
        self.assertTrue(is_ok)

    def test_calibrate_normal(self):
        is_ok = True
        try:
            self.normal_camera.calibrate("tests/assets/left_camera_calibration", show=False, fish_eye=False, export_file=False)
        except:
            is_ok = False
        self.assertTrue(is_ok)
    
    def test_get_fisheye_parameters(self):
        is_ok = True
        output_path = "{}_parameters.xml".format(self.fisheye_camera.id)
        self.fisheye_camera.calibrate("tests/assets/right_camera_calibration", show=False, export_file=True)
        try:
            self.fisheye_camera.get_parameters(output_path)
        except:
            is_ok = False
        os.remove(output_path)
        self.assertTrue(is_ok)

    def test_get_normal_parameters(self):
        is_ok = True
        output_path = "{}_parameters.xml".format(self.fisheye_camera.id)
        self.fisheye_camera.calibrate("tests/assets/right_camera_calibration", show=False, export_file=True)
        try:
            self.fisheye_camera.get_parameters(output_path)
        except:
            is_ok = False
        os.remove(output_path)
        self.assertTrue(is_ok)
    
    def test_get_parameters_with_bad_path(self):
        is_bad_path = False
        output_path = "{}_parameters.xml".format(self.fisheye_camera.id)
        self.fisheye_camera.calibrate("tests/assets/right_camera_calibration", show=False, export_file=True)
        try:
            self.fisheye_camera.get_parameters("bad_path_example.xml")
        except OSError:
            is_bad_path = True
        os.remove(output_path)
        self.assertTrue(is_bad_path)
        
if __name__ == '__main__':
    unittest.main()