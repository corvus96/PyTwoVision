import glob
import os
import cv2 as cv
import numpy as np 
import requests
import re
import errno

class Camera:
    
    def __init__(self, id, source):
        """Build SSD model given a backbone
        Arguments:
            id (str): An identifier to our camera.
            source (int | str): When you use webcam you need to put (int) 0,
            if you want to use videos or streaming, you will need to put his URL or path.
        Returns:
            A model of the real world camera to use in stereo problems 
        """
        self.id = id
        self.source = source
        # define type of source
        regex = '((http|https)://)(www.)?' + '[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]' + '{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)'
        regular_exp = re.compile(regex)
        if type(self.source) is str:
            if re.search(regular_exp, self.source):
                self.type_source = 'stream'
            else:
                self.type_source = 'other'
        else:
            self.type_source = 'webcam'

    def take_photos(self, num_photos=15, save_dir="images", prefix_name="photo"):
        """ A simple way to take photos by console and save in a folder
        Arguments:
            num_photos (int): Number of photos to take
            save_dir (str): Directory name where the photos will be saved
            prefix_name (str): A prefix for the names of the photos
        """
        # initial directory
        cwd = os.getcwd()
        # make directory
        try:
            os.mkdir(save_dir)
            os.chdir(save_dir)
        except OSError as error:
            print(error)   
            if error.errno == errno.EEXIST:
                os.chdir(save_dir)


        if self.type_source == 'other' or 'webcam':
            cap = cv.VideoCapture(self.source)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()

        for i in range(num_photos):
            print("Please press enter to take a photo")
            while True:
                if self.type_source == 'stream':
                    img_resp = requests.get(self.source)
                    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                    frame = cv.imdecode(img_arr, -1)
                    cv.imshow("streaming: {}".format(self.id), frame)
                elif self.type_source == 'other' or 'webcam':
                    _, frame = cap.read()
                    cv.imshow("webcam: {}".format(self.id), frame)
                input_key = cv.waitKey(1)
                
                if input_key == 32:
                    #space pressed
                    cv.imwrite('{}_{}.png'.format(prefix_name, i + 1), frame)
                    print("{} photos left".format(num_photos - i - 1))
                    break
                if input_key == 27:
                    #esc pressed
                    print("Escape hit, closing...")
                    break
            if input_key == 27:
                #esc pressed
                break
        
        # restore to initial directory
        os.chdir(cwd)

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
            ret, matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
        except OSError:
            print("Could not find any image in {}".format(images_path))

cam1 = Camera("cam1", 0)
cam1.take_photos()