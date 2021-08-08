import glob
import os
import cv2 as cv
import numpy as np 

class Camera:
    def __init__(self, identifier):
        self.id = identifier
        self.camera_matrix = []
        self.ret = None
        self.rvecs = None
        self.tvecs = None
        self.dist_coeffs = None
    
    def take_photos(self, num_photos=15):
        stop = False
        event = False
        for i in range(num_photos):
            if stop:
                break
            while event is False:
                input_key = input("Please press space to take a photo\n")
                if input_key == '':
                    print("{} photos left".format(num_photos - i - 1))
                    break

    def calibrate(self, images_path='', pattern_type='chessboard', pattern_size=(7,6)):
        # adjust image path
        images_path, _ = os.path.splitext(images_path)
        # available formats
        formats = ['*.svg', '*.png', '*.jpg', '*.bmp', '*.jpeg', '*.raw']
        # get files names
        files_grabbed = [glob.glob((images_path + '/' + e)  if len(images_path) > 0 else e) for e in formats]
        # flatting files list
        images = [item for sublist in files_grabbed for item in sublist]

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        try:
            if len(images) == 0:
                raise OSError
            for fname in images:
                img = cv.imread(fname)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                if pattern_type == 'chessboard':
                    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners)
            self.ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
        except OSError:
            print("Could not find any image in {}".format(images_path))

cam1 = Camera("cam1")
cam1.take_photos()