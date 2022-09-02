import os
import glob
import numpy as np
import cv2 as cv 
import requests
import time

from py2vision.stereo.stereo_builder import StereoSystemBuilder
from py2vision.stereo.match_method import Matcher
from py2vision.input_output.camera import Camera
from py2vision.utils.draw import draw_lines
from py2vision.compute.error_compute import re_projection_error


class StandardStereo:
    """An emulation of a real world stereo system with his relevant parameters, like fundamental matrix, rotation matrix, translation matrix and esential matrix. 

    Attributes:
        camL: left camera instance.
        camR: right camera instance.
        fish_eye: A boolean if is true it will calibrate with cv2.fisheye.calibrate if no it'll use normal calibration, fish eye is recomended when cameras has an field of view > 160.
    """
    def __init__(self, cam_left: Camera, cam_right: Camera, fish_eye=True):
        if type(fish_eye) != bool:
            raise ValueError("fish_eye has to be a boolean") 
        self.camL = cam_left
        self.camR = cam_right
        self.fish_eye = fish_eye
    
    def take_dual_photos(self, num_photos=15, save_dir_left="images_left", save_dir_right="images_right", prefix_name="photo"):
        """ A simple way to take photos in console and save in a folder.
        
        Args:
            num_photos: number of photos to take, 15 by default.
            save_dir_left: directory name where the left photos will be saved.
            save_dir_right: directory name where the right photos will be saved.
            prefix_name: A prefix for the names of the photos.

        Raises:
            OSError: if folder exist.
        """
        # initial directory
        cwd = os.getcwd()
        # make directory
        try:
            os.mkdir(save_dir_left)
        except OSError as error:
            print(error)   
        try:
            os.mkdir(save_dir_right)
        except OSError as error:
            print(error)   


        if self.camL.type_source == 'other' or self.camL.type_source == 'webcam':
            capL = cv.VideoCapture(self.camL.source)
            if not capL.isOpened():
                print("Cannot open left camera")
                exit()
        if self.camR.type_source == 'other' or self.camR.type_source == 'webcam':
            capR = cv.VideoCapture(self.camR.source)
            if not capR.isOpened():
                print("Cannot open right camera")
                exit()

        for i in range(num_photos):
            print("Please press enter to take a photo")
            while True:
                if self.camL.type_source == 'stream' and self.camR.type_source == 'stream':
                    img_respL = requests.get(self.camL.source)
                    img_arrL = np.array(bytearray(img_respL.content), dtype=np.uint8)
                    frameL = cv.imdecode(img_arrL, -1)
                    img_respR = requests.get(self.camR.source)
                    img_arrR = np.array(bytearray(img_respR.content), dtype=np.uint8)
                    frameR = cv.imdecode(img_arrR, -1)
                    cv.namedWindow("streaming: {}".format(self.camL.id), cv.WINDOW_NORMAL)
                    cv.imshow("streaming: {}".format(self.camL.id), frameL)
                    cv.namedWindow("streaming: {}".format(self.camR.id), cv.WINDOW_NORMAL)
                    cv.imshow("streaming: {}".format(self.camR.id), frameR)
                elif (self.camL.type_source == 'other' or 'webcam') and self.camR.type_source == 'stream':
                    _, frameL = capL.read()
                    img_respR = requests.get(self.camR.source)
                    img_arrR = np.array(bytearray(img_respR.content), dtype=np.uint8)
                    frameR = cv.imdecode(img_arrR, -1)
                    cv.imshow("webcam: {}".format(self.camL.id), frameL)
                    cv.namedWindow("streaming: {}".format(self.camR.id), cv.WINDOW_NORMAL)
                    cv.imshow("streaming: {}".format(self.camR.id), frameR)
                elif (self.camR.type_source == 'other' or 'webcam') and self.camL.type_source == 'stream':
                    _, frameR = capR.read()
                    img_respL = requests.get(self.camL.source)
                    img_arrL = np.array(bytearray(img_respL.content), dtype=np.uint8)
                    frameL = cv.imdecode(img_arrL, -1)
                    cv.imshow("webcam: {}".format(self.camR.id), frameR)
                    cv.namedWindow("streaming: {}".format(self.camL.id), cv.WINDOW_NORMAL)
                    cv.imshow("streaming: {}".format(self.camL.id), frameL)
                else:
                    _, frameL = capL.read()
                    _, frameR = capR.read()
                    cv.imshow("webcam: {}".format(self.camL.id), frameL)
                    cv.imshow("webcam: {}".format(self.camR.id), frameR)
                input_key = cv.waitKey(1)
                
                if input_key == 32:
                    #space pressed
                    os.chdir(save_dir_left)
                    cv.imwrite('{}_{}.png'.format(prefix_name, i + 1), frameL)
                    print("saved as {}_{}.png".format(prefix_name, i + 1))
                    os.chdir(cwd)
                    os.chdir(save_dir_right)
                    cv.imwrite('{}_{}.png'.format(prefix_name, i + 1), frameR)
                    print("saved as {}_{}.png".format(prefix_name, i + 1))
                    os.chdir(cwd)
                    print("{} photos left".format(num_photos - i - 1))
                    break
                if input_key == 27:
                    #esc pressed
                    print("Escape hit, closing...")
                    break
            if input_key == 27:
                #esc pressed
                break
        cv.destroyAllWindows()
        # restore to initial directory
        os.chdir(cwd)

    def calibrate(self, images_left_path, images_right_path, pattern_type='chessboard', pattern_size=(8,5), show=True):
        """ Compute intrinsics and extrinsics parameters for two cameras at once.
        Note: if you want a good calibration your images in 'images_left_path' and 'images_right_path' have to follow a format for example: example/left_images/photo_1.jpg and example/right_images/photo_1.jpg the number is important because calibrate method order the pair of images with sorted method.
        
        Args:
            images_left_path: folder where is saved left calibration pattern photos.
            images_right_path: folder where is saved right calibration pattern photos.
            pattern_type: It can be "circles" pattern or "chessboard" pattern (default).
            pattern_size: If pattern_type is "chessboard"  this the Number of inner corners per a chessboard row and column. But If pattern_type is "circles" this will be the number of circles per row and column. 
            show: if is true it show corners or centers found by calibration algorithm  at each iteration.
        
        Returns: 
            Three floats, The first is the reprojection error in left side,  the another one is right side
            and the third one is rms error (these have to be less than 1).

        Raises:
            OSError: If didn't find photos on images_left_path or images_right_path folders.
        """
        valid_patterns = ["chessboard", "circles"]
        if pattern_type not in valid_patterns:
            raise ValueError("pattern_type only can be 'chessboard' or 'circles'") 
        # adjust image path
        images_left, _ = os.path.splitext(images_left_path)
        images_right, _ = os.path.splitext(images_right_path)
        # available formats
        formats = ['*.svg', '*.png', '*.jpg', '*.bmp', '*.jpeg', '*.raw']
        # get files names
        filesL = [glob.glob((images_left + '/' + e)  if len(images_left) > 0 else e) for e in formats]
        filesR = [glob.glob((images_right + '/' + e)  if len(images_right) > 0 else e) for e in formats]
        # flatting files list
        imagesL = [item for sublist in filesL for item in sublist]
        imagesR = [item for sublist in filesR for item in sublist]
        imagesL = sorted(imagesL)
        imagesR = sorted(imagesR)
        # termination criteria
        if self.fish_eye == False:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        else:
            criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        if self.fish_eye == False:
            objp = np.zeros((pattern_size[1]*pattern_size[0],3), np.float32)
            objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        else:
            objp = np.zeros((pattern_size[1]*pattern_size[0], 1, 3), np.float32)
            objp[:,0, :2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpointsL = [] # 2d points in image plane.
        imgpointsR = [] # 2d points in image plane.
        try:
            if len(imagesL) == 0 or len(imagesR) == 0:
                raise OSError
            for imgLeft, imgRight in zip(imagesL, imagesR):
                imgL = cv.imread(imgLeft)
                imgR = cv.imread(imgRight)
                grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
                grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                if pattern_type == 'chessboard' and self.fish_eye == False:
                    retL, cornersL = cv.findChessboardCorners(grayL, pattern_size, None)
                    retR, cornersR = cv.findChessboardCorners(grayR, pattern_size, None)
                elif pattern_type == 'circles' and self.fish_eye == False:
                    retL, cornersL = cv.findCirclesGrid(grayL, pattern_size, None)
                    retR, cornersR = cv.findCirclesGrid(grayR, pattern_size, None)
                elif pattern_type == 'chessboard' and self.fish_eye == True:
                    retL, cornersL = cv.findChessboardCorners(grayL, pattern_size, flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
                    retR, cornersR = cv.findChessboardCorners(grayR, pattern_size, flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
                elif pattern_type == 'circles' and self.fish_eye == True:
                    retL, cornersL = cv.findCirclesGrid(grayL, pattern_size, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
                    retR, cornersR = cv.findCirclesGrid(grayR, pattern_size, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

                # If found, add object points, image points (after refining them)
                if retL and retR == True:

                    objpoints.append(objp)

                    if self.fish_eye == False:   
                        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                    else:
                        cornersL = cv.cornerSubPix(grayL, cornersL, (3,3), (-1,-1), criteria)
                        cornersR = cv.cornerSubPix(grayR, cornersR, (3,3), (-1,-1), criteria)
                    imgpointsL.append(cornersL)
                    imgpointsR.append(cornersR)

                if show:
                    # Draw and display the corners
                    cv.drawChessboardCorners(imgL, pattern_size, cornersL, retL)
                    cv.namedWindow(imgLeft, cv.WINDOW_NORMAL)
                    cv.imshow(imgLeft, imgL)
                    cv.drawChessboardCorners(imgR, pattern_size, cornersR, retR)
                    cv.namedWindow(imgRight, cv.WINDOW_NORMAL)
                    cv.imshow(imgRight, imgR)
                    cv.waitKey(6000)
                    cv.destroyAllWindows()
            
            if self.fish_eye == False:
                flags = 0
                flags |= cv.CALIB_FIX_INTRINSIC
                # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
                # Hence intrinsic parameters are the same 

                criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            else:
                flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
                criteria_stereo = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                k = np.zeros((3, 3))
                d = np.zeros((4, 1))
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
            
            try:
                # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
                print("Calibrating estereo...")
                if self.fish_eye == False:
                    rms_error, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, self.rot, self.trans, self.e_matrix, self.f_matrix, _ = cv.stereoCalibrateExtended(objpoints, imgpointsL, imgpointsR, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], criteria_stereo, flags)
                else:
                    rms_error, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, self.rot, self.trans, = cv.fisheye.stereoCalibrate(objpoints, imgpointsL, imgpointsR, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], None, None, cv.CALIB_FIX_INTRINSIC, criteria_stereo)
                errorL = re_projection_error(objpoints, self.camL.rvecs, self.camL.tvecs, self.camL.matrix, imgpointsL, self.camL.dist_coeffs)
                print("Left camera error {}".format(errorL))
                errorR = re_projection_error(objpoints, self.camR.rvecs, self.camR.tvecs, self.camR.matrix, imgpointsR, self.camR.dist_coeffs)
                print("Right camera error {}".format(errorR))
                print("system calibrated")
                
            except AttributeError:
                print("Camera parameters hasn't fixed")
                print("Fixing left camera...")
                if self.fish_eye == False:
                    errorL, self.camL.matrix, self.camL.dist_coeffs, self.camL.rvecs, self.camL.tvecs = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
                    heightL, widthL = imgL.shape[:2]
                    self.camL.matrix, self.camL.roi = cv.getOptimalNewCameraMatrix(self.camL.matrix, self.camL.dist_coeffs, (widthL, heightL), 1, (widthL, heightL))
                else:
                    errorL, self.camL.matrix, self.camL.dist_coeffs, self.camL.rvecs, self.camL.tvecs = cv.fisheye.calibrate(objpoints, imgpointsL, grayL.shape[::-1], k, d, rvecs, tvecs, flags, (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
                print("Left camera error {}".format(errorL))
                print("Fixing Right camera...")
                if  self.fish_eye == False:
                    errorR, self.camR.matrix, self.camR.dist_coeffs, self.camR.rvecs, self.camR.tvecs = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
                    heightR, widthR = imgL.shape[:2]
                    self.camR.matrix, self.camR.roi = cv.getOptimalNewCameraMatrix(self.camR.matrix, self.camR.dist_coeffs, (widthR, heightR), 1, (widthR, heightR))
                else:
                    errorR, self.camR.matrix, self.camR.dist_coeffs, self.camR.rvecs, self.camR.tvecs = cv.fisheye.calibrate(objpoints, imgpointsR, grayR.shape[::-1], k, d, rvecs, tvecs, flags, (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
                print("Right camera error {}".format(errorR))
                print("Calibrating estereo...")
                # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
                if self.fish_eye == False:
                    self.ret_stereo, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, self.rot, self.trans, self.e_matrix, self.f_matrix, rms_error = cv.stereoCalibrateExtended(objpoints, imgpointsL, imgpointsR, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], criteria_stereo, flags)
                else:
                    rms_error, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, self.rot, self.trans, = cv.fisheye.stereoCalibrate(objpoints, imgpointsL, imgpointsR, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], None, None, cv.CALIB_FIX_INTRINSIC, criteria_stereo)
                print("system calibrated")
        except OSError:
            if len(imagesL) == 0:
                print("Could not find any image in {}".format(images_left_path))
            if len(imagesR) == 0:
                print("Could not find any image in {}".format(images_right_path))

        return errorL, errorR, rms_error

    def rectify(self, image_left__dims, image_right_dims, export_file_name="stereoMap", alpha=1, output_size=(0,0), export_file=True):
        """ Compute stereo rectification maps and export left and right stereo maps in xml format. Note: you will need to calibrate first
        
        Args: 
            image_left__dims: a tuple or list with left image (width, height).
            image_right_dims: a tuple or list with right image (width, height).
            export_file_name: personalize the name of output file.
            alpha: free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 (default) means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases (Only apply for no fish eye cameras). 
            output_size: New image resolution after rectification. When (0,0) is passed (default), it is set to the original image Size. Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion.
            export_file: if is true this method will save the parameters in an xml with the name in export_file_name. 
        
        Raises: 
            AttributeError: If haven't calibrated before, you need all parameters of calibrate() method.
         """
        if len(image_left__dims) != 2 or len(image_right_dims) != 2:
            raise ValueError("image_left_dims and image_right_dims must have a length of 2")
        if self.fish_eye:
            r2 = np.zeros([3,3])
            p1 = np.zeros([3,4])
            p2 = np.zeros([3,4])
            q = np.zeros([4,4])
        try:
            if self.fish_eye == False:
                rectL, rectR, projMatrixL, projMatrixR, self.Q, self.camL.roi, self.camR.roi = cv.stereoRectify(self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, image_left__dims, self.rot, self.trans, alpha, output_size, flags=cv.CALIB_ZERO_DISPARITY)

                self.stereoMapL = cv.initUndistortRectifyMap(self.camL.matrix, self.camL.dist_coeffs, rectL, projMatrixL, image_left__dims, cv.CV_16SC2)
                self.stereoMapR = cv.initUndistortRectifyMap(self.camR.matrix, self.camR.dist_coeffs, rectR, projMatrixR, image_right_dims, cv.CV_16SC2)
            else:
                rectL, rectR, projMatrixL, projMatrixR, self.Q = cv.fisheye.stereoRectify(self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, image_left__dims, self.rot, self.trans, 0, r2, p1, p2, q, cv.CALIB_ZERO_DISPARITY, output_size, 0, 0)

                self.stereoMapL = cv.fisheye.initUndistortRectifyMap(self.camL.matrix, self.camL.dist_coeffs, rectL, projMatrixL, image_left__dims, cv.CV_16SC2)
                self.stereoMapR = cv.fisheye.initUndistortRectifyMap(self.camR.matrix, self.camR.dist_coeffs, rectR, projMatrixR, image_right_dims, cv.CV_16SC2)
            if export_file:
                print("Saving parameters!")
                cv_file = cv.FileStorage('{}.xml'.format(export_file_name), cv.FILE_STORAGE_WRITE)

                cv_file.write('stereoMapL_x', self.stereoMapL[0])
                cv_file.write('stereoMapL_y', self.stereoMapL[1])
                cv_file.write('stereoMapR_x', self.stereoMapR[0])
                cv_file.write('stereoMapR_y', self.stereoMapR[1])

                cv_file.release()
        except AttributeError:
            print("Please calibrate first!")
    
    def get_stereo_maps(self, path):
        """ Loads stereo maps parameters from a xml file
        
        Args: 
            path: path where is saved xml file. (Note: if you don't have stereo maps you can use calibrate method first, then use rectify method and pass true in export file argument)
         """
        # file storage read
        cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
        if not cv_file.isOpened():
            raise OSError("Looks like {} doesn't exist!".format(path))
        temp_stereoMapL = []
        temp_stereoMapR = []
        temp_stereoMapL.append(cv_file.getNode('stereoMapL_x').mat())
        temp_stereoMapL.append(cv_file.getNode('stereoMapL_y').mat())
        temp_stereoMapR.append(cv_file.getNode('stereoMapR_x').mat())
        temp_stereoMapR.append(cv_file.getNode('stereoMapR_y').mat())
        # copy to class attributes
        self.stereoMapL = temp_stereoMapL
        self.stereoMapR = temp_stereoMapR
        cv_file.release()
    
    def print_parameters(self, one_camera_parameters=['matrix', 'dist_coeffs'], stereo_parameters=['Q', 'rot', 'trans', 'e_matrix', 'f_matrix']):
        """ Print required individual camera parameters and stereo parameters already defined.

        Args: 
            one_camera_parameters: All elements in list  will be printed, they match with cameras parameters.
            stereo_parameters: All elements in list  will be printed, they match with stereo parameters. 
         """
        np.set_printoptions(suppress=True)
        print("Left camera:")
        for attr in dir(self.camL):
            if attr in one_camera_parameters:
                print("{} : {}".format(attr, getattr(self.camL, attr)))
        
        print("\n")
        print("Right camera:")
        for attr in dir(self.camR):
            if attr in one_camera_parameters:
                print("{} : {}".format(attr, getattr(self.camR, attr)))
        print("\n")
        print("Stereo:")
        for attr in dir(self):
            if attr in stereo_parameters:
                print("{} : {}".format(attr, getattr(self, attr)))
        np.set_printoptions(suppress=False)

class StandardStereoBuilder(StereoSystemBuilder):
    """
    Implement methods to get depth using OpenCV matchers like SGBM or BM.
    
    Attributes:
        camL: left camera instance.
        camR: right camera instance.
        stereo_maps_path: an xml with Stereo rectification maps
    """

    def __init__(self, cam_left: Camera, cam_right: Camera, stereo_maps_path) -> None:
        """
        A fresh builder instance should contain a blank product object, which is used in further assembly.
        """
        self.reset(cam_left, cam_right, stereo_maps_path)

    def reset(self, cam_left: Camera, cam_right: Camera, stereo_maps_path) -> None:
        """
        Initialitation of product
        """
        self._product = StandardStereo(cam_left, cam_right)
        self._product.get_stereo_maps(stereo_maps_path)

    def get_product(self):
        """
        Return: StandardStereo builded object
        """
        return self._product

    def pre_process(self, frameL, frameR, downsample=2):
        """ First, it transform from BGR to gray, next apply rectification and finally apply pyramid subsampling.
        
        Args: 
            frameL: it's the left frame
            frameR: it's the right frame
            downsample: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor just can be 2, 4, 8, 16, 32, 64. If downsample factor is 1 or None or False won't apply downsampling.
        
        Returns:
            Both frames to apply stereo correspondence or apply more processing to the images.
        """
        if downsample not in (2**p for p in range(1, 7)):
            if downsample in [1, None, False]:
                n_downsamples = 0
            else:
                raise ValueError("downsample only can be 2, 4, 8, 16, 32, 64")
        else:
            n_downsamples = [2**p for p in range(1, 7)].index(downsample)
            n_downsamples += 1
        
        img_left_gray = cv.cvtColor(frameL,cv.COLOR_BGR2GRAY)
        img_right_gray = cv.cvtColor(frameR,cv.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image
        rectified_left = cv.remap(img_left_gray, self._product.stereoMapL[0], self._product.stereoMapL[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        # Applying stereo image rectification on the right image
        rectified_right = cv.remap(img_right_gray, self._product.stereoMapR[0], self._product.stereoMapR[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        if n_downsamples > 0:
            for i in range(n_downsamples):
                rectified_left = cv.pyrDown(rectified_left)
                rectified_right = cv.pyrDown(rectified_right)
        
        return rectified_left, rectified_right
    
    def find_epilines(self, frameL, frameR):
        """ Draw epilines to both frames.
        
        Args: 
            frameL: it's the left frame.
            frameR: it's the right frame.
        
        Returns:
            Two elements, the left and right frame with epilines
        """
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(frameL,None)
        kp2, des2 = sift.detectAndCompute(frameR,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5, _ = draw_lines(frameL, frameR,lines1,pts1,pts2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3, _ = draw_lines(frameR, frameL,lines2,pts2,pts1)
        return img5, img3

    def match(self, frameL, frameR, matcher: Matcher, metrics=True):
        """ Apply stereo SGBM.
        
        Args: 
            frameL: it's the left frame.
            frameR: it's the right frame.
            metrics: a boolean, if is true print by console the time of execution of correspondence step.

        Returns:
            left and right disparity maps and even matcher instance.
        """
        left_matcher = matcher.match()
        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
        if metrics:
            print('computing disparity...') 
            matching_time = time.time()
        left_disp = left_matcher.compute(frameL, frameR)
        right_disp = right_matcher.compute(frameR, frameL)
        if metrics:
            print("matching time: {} s".format(time.time() - matching_time))
        return left_disp, right_disp, left_matcher
         
    
    def post_process(self, frameL, left_disp, right_disp, matcher, lmbda=8000, sigma=1.5, metrics=True):
        """ Apply wls filter in disparity maps to smooth contours.
        
        Args: 
            frameL: it's the left frame.
            left_disp: it's the left disparity frame.
            right_disp: it's the right disparity frame.
            matcher: it's the matcher instance.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step.
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step
            metrics: if is true print by console the time of execution of post process step.

        Returns:
            Improved disparity map
        """
        wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher)
        wls_filter.setLambda(lmbda);
        wls_filter.setSigmaColor(sigma);
        if metrics:
            postprocessing_time = time.time()
        filtered_disp = wls_filter.filter(left_disp, frameL, disparity_map_right=right_disp);
        if metrics:
            print("postprocessing time: {} s".format(time.time() - postprocessing_time))
        return filtered_disp

    def estimate_disparity_colormap(self, disparity):
        """ It converts disparity maps with a shape of (w, h, 1) to  (w, h, 3).

        Args: 
            disparity: a disparity map with a shape of (w, h, 1).

        Returns:
            disparity with color
        """
        return cv.applyColorMap((disparity).astype(np.uint8), cv.COLORMAP_JET)
    
    def estimate_depth_map(self, disparity, Q, frame, metrics=True):
        """ Convert disparity map to depth map.

        Args:
            frame: left or right frame.
            disparity: a disparity map with a shape of (w, h, 1)
            Q: a 4x4 array with the following structure, [[1 0   0          -cx     ][0 1   0          -cy     ][0 0   0           f      ][0 0 -1/Tx (cx - cx')/Tx ]] cx: is the principal point x in left image, cx': is the principal point x in right image, cy: is the principal point y in left image, f: is the focal lenth in left image, Tx: The x coordinate in Translation matrix.  
            metrics: a boolean, if is true print by console the time of execution of depth map step.

        Returns:
            reprojected points image and a depth map in RGB
        """
        if metrics:
            get_3D_points_time = time.time()
        disparity = disparity.astype(np.float32) / 16.0
        points = cv.reprojectImageTo3D(disparity, Q)
        colors = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = disparity > disparity.min()
        out_points = points[mask]
        out_colors = colors[mask]
        if metrics:
            print("3D points time: {} s".format(time.time() - get_3D_points_time))
        return out_points, out_colors

    def estimate_3D_points(self, points, disparity, Q):
        """ Convert image plane points to homogeneous 3d points.

        Args:
            points: contains the (x, y) coordinates in image plane to convert to (X, Y, Z).
            disparity: a disparity map with a shape of (w, h, 1)
            Q: a 4x4 array with the following structure, [[1 0   0          -cx     ][0 1   0          -cy     ][0 0   0           f      ][0 0 -1/Tx (cx - cx')/Tx ]] cx: is the principal point x in left image, cx': is the principal point x in right image, cy: is the principal point y in left image, f: is the focal lenth in left image, Tx: The x coordinate in Translation matrix.  

        Returns:
            An array of points in 3D homogeneous coordinates (X, Y, Z, W).
        """
        points = np.array(points)
        disparity = np.array(disparity)
        homogeneous_3D_points = []
        for point in points:
            point_xy_disp = np.append(point, [disparity[point[1]][point[0]], 1], axis=0)
            projected_point = np.matmul(Q, point_xy_disp)
            homogeneous_3D_points.append(projected_point)
        return homogeneous_3D_points
