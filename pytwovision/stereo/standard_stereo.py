from _typeshed import Self
import os
import glob
import numpy as np
import cv2 as cv 

from stereo.stereo_builder import StereoSystemBuilder
from stereo.stereo_builder import StereoController 
from typing import Any
from input_output.camera import Camera




class StandardStereo():
    """An emulation of a real world stereo system with his relevant parameters, like 
        fundamental matrix, rotation matrix, translation matrix and esential matrix. 
    Arguments:
        camL (Obj): left camera instance.
        camR (Obj): right camera instance.
    """
    def __init__(self, cam_left: Camera, cam_right: Camera) -> None:
        self.camL = cam_left
        self.camR = cam_right
    
    def calibrate(self, images_left_path, images_right_path, pattern_type='chessboard', pattern_size=(7,6)):
        """ Compute stereo parameters like a fundamental matrix, 
        rotation and traslation matrix and esential matrix.
        Arguments:
            images_left_path (str): folder where is saved left calibration pattern photos.
            images_right_path (str): folder where is saved right calibration pattern photos.
            pattern_type (str): It can be "circles" pattern or "chessboard" pattern (default).
            pattern_sizev(tuple): If pattern_type is "chessboard"  this the Number of inner corners per a chessboard row and column. But If pattern_type is "circles" this will be the number of circles per row and column. 
        Raises:
            OSError: If didn't find photos on images_left_path or images_right_path folders.
        """
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
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((pattern_size[1]*pattern_size[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
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
                if pattern_type == 'chessboard':
                    retL, cornersL = cv.findChessboardCorners(grayL, pattern_size, None)
                    retR, cornersR = cv.findChessboardCorners(grayR, pattern_size, None)
                elif pattern_type == 'circles':
                    retL, cornersL = cv.findCirclesGrid(grayL, pattern_size, None)
                    retR, cornersR = cv.findCirclesGrid(grayR, pattern_size, None)

                # If found, add object points, image points (after refining them)
                if retL and retR == True:

                    objpoints.append(objp)

                    cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                    imgpointsL.append(cornersL)

                    cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                    imgpointsR.append(cornersR)

                    # Draw and display the corners
                    # cv.drawChessboardCorners(imgL, pattern_size, cornersL, retL)
                    # cv.namedWindow('img left', cv.WINDOW_NORMAL)
                    # cv.imshow('img left', imgL)
                    # cv.drawChessboardCorners(imgR, pattern_size, cornersR, retR)
                    # cv.namedWindow('img right', cv.WINDOW_NORMAL)
                    # cv.imshow('img right', imgR)
                    # cv.waitKey(1000)
            
            flags = 0
            flags |= cv.CALIB_FIX_INTRINSIC
            # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
            # Hence intrinsic parameters are the same 

            criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            try:
                # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
                print("Calibrating estereo...")
                self.ret_stereo, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, self.rot, self.trans, self.e_matrix, self.f_matrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], criteria_stereo, flags)
                print("system calibrated")
            except AttributeError:
                print("Camera parameters hasn't fixed")
                print("Fixing left camera...")
                self.camL.ret, self.camL.matrix, self.camL.dist_coeffs, self.camL.rvecs, self.camL.tvecs = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
                heightL, widthL = imgL.shape[:2]
                self.camL.matrix, self.camL.roi = cv.getOptimalNewCameraMatrix(self.camL.matrix, self.camL.dist_coeffs, (widthL, heightL), 1, (widthL, heightL))
                print("Fixing Right camera...")
                self.camR.ret, self.camR.matrix, self.camR.dist_coeffs, self.camR.rvecs, self.camR.tvecs = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
                heightR, widthR = imgL.shape[:2]
                self.camR.matrix, self.camR.roi = cv.getOptimalNewCameraMatrix(self.camR.matrix, self.camR.dist_coeffs, (widthR, heightR), 1, (widthR, heightR))
                print("Calibrating estereo...")
                # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
                self.ret_stereo, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, self.rot, self.trans, self.e_matrix, self.f_matrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], criteria_stereo, flags)
                print("system calibrated")
            
        except OSError:
            if len(imagesL) == 0:
                print("Could not find any image in {}".format(images_left_path))
            if len(imagesR) == 0:
                print("Could not find any image in {}".format(images_right_path))

    def rectify(self, image_left_path, image_right_path, export_file_name="stereoMap", alpha=1, output_size=(0,0), export_file=True):
        """ Compute stereo rectification maps and export left and right stereo maps in xml format.
            Note: you will need to calibrate first
        Arguments: 
            image_left_path (str): The path of an example calibration left image to get his dimensions.
            image_right_path (str): The path of an example calibration right image to get his dimensions.
            export_file_name (str): personalize the name of output file.
            alpha (double): Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 (default) means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases.
            output_size: New image resolution after rectification. When (0,0) is passed (default), it is set to the original image Size . Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion.
        Raises: 
            AttributeError: If haven't calibrated before, you need all parameters of calibrate() method.
         """
        imgL = cv.imread(image_left_path)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        imgR = cv.imread(image_right_path)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        try:
            rectL, rectR, projMatrixL, projMatrixR, Q, self.camL.roi, self.camR.roi = cv.stereoRectify(self.camL.matrix, self.camL.dist_coeffs, self.camR.matrix, self.camR.dist_coeffs, grayL.shape[::-1], self.rot, self.trans, alpha, output_size)

            self.stereoMapL = cv.initUndistortRectifyMap(self.camL.matrix, self.camL.dist_coeffs, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
            self.stereoMapR = cv.initUndistortRectifyMap(self.camR.matrix, self.camR.dist_coeffs, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)
            
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
        Arguments: 
            path (str): path where is saved xml file. (Note: if you don't 
            have stereo maps you can use calibrate method first,
            then use rectify method and pass true in export file argument)
         """
        # file storage read
        cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
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
    
    def print_parameters(self, one_camera_parameters=['matrix', 'dist_coeffs', 'roi'], stereo_parameters=['ret_stereo', 'rot', 'trans', 'e_matrix', 'f_matrix']):
        """ Print required individual camera parameters and stereo parameters already defined
        Arguments: 
            one_camera_parameters (list): All elements in list  will be printed, 
            they match with cameras parameters.
            stereo_parameters (list): All elements in list  will be printed, 
            they match with stereo parameters. 
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
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps.
    """

    def __init__(self, cam_left: Camera, cam_right: Camera, stereo_maps_path) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
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

    def pre_process(self, frameL, frameR) -> None:
        img_left_gray = cv.cvtColor(frameL,cv.COLOR_BGR2GRAY)
        img_right_gray = cv.cvtColor(frameR,cv.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image
        retify_left = cv.remap(img_left_gray, self._product.stereoMapL[0], self._product.stereoMapL[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        # Applying stereo image rectification on the right image
        retify_right = cv.remap(img_right_gray, self._product.stereoMapR[0], self._product.stereoMapR[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        
        return retify_left, retify_right

if __name__ == "__main__":
    """
    The client code creates a builder object, passes it to the director and then
    initiates the construction process. The end result is retrieved from the
    builder object.
    """
    # Test without previous calibration
    # cam1 = Camera("xiaomi_redmi_note_8_pro", 'http://192.168.0.107:8080/shot.jpg')
    # cam2 = Camera("xiaomi_redmi_9T", 'http://192.168.0.109:8080/shot.jpg')
    # stereo = StandardStereo(cam1, cam2)
    # stereo.calibrate("/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/left_camera_xiaomi_redmi_note_8_pro", "/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/right_camera_xiaomi_redmi_9T", pattern_size=(8, 5))

    # Test with previous calibration
    cam1 = Camera("xiaomi_redmi_note_8_pro", 'http://192.168.0.107:8080/shot.jpg')
    cam2 = Camera("xiaomi_redmi_9T", 'http://192.168.0.109:8080/shot.jpg')
    cam1.get_parameters("/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/xiaomi_redmi_note_8_pro_parameters.xml")
    cam2.get_parameters("/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/xiaomi_redmi_9T_parameters.xml")
    stereo = StandardStereo(cam1, cam2)
    stereo.calibrate("/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/left_camera_xiaomi_redmi_note_8_pro", "/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/right_camera_xiaomi_redmi_9T", pattern_size=(8, 5))
    stereo.rectify("/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/left_camera_xiaomi_redmi_note_8_pro/photo_1.png",  "/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/right_camera_xiaomi_redmi_9T/photo_1.png", "different_left_right")
    print("Not equals images test parameters:")
    stereo.print_parameters()
    stereo.rectify("/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/left_one_photo/photo_1.png",  "/home/corvus96/Documentos/github/PyTwoVision/pytwovision/input_output/right_one_photo/photo_1.png", "equal_left_right")
    stereo.print_parameters()

    # director = StereoController()
    # builder = StandardStereoBuilder()
    # director.builder = builder

    # print("Standard basic product: ")
    # director.build_minimal_viable_product()
    # builder.product.list_parts()

    # print("\n")

    # print("Standard full featured product: ")
    # director.build_full_featured_product()
    # builder.product.list_parts()

    # print("\n")

    # # Remember, the Builder pattern can be used without a Director class.
    # print("Custom product: ")
    # builder.produce_part_a()
    # builder.produce_part_b()
    # builder.product.list_parts()