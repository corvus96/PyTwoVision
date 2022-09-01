import glob
import os
import cv2 as cv
import numpy as np 
import requests
import re
import errno

from pytwovision.image_process.frame_decorator import Frame
from pytwovision.image_process.resize import Resize
from pytwovision.image_process.rotate import Rotate

class Camera():
    """An emulation of a real world camera with his relevant parameters, like camera matrix, extrinsics and intrinsics parameters. 

    Args:
        id: A string to identify our camera.
        source: When you use webcam you need to put (int) 0, if you want to use videos or streaming, you will need to put his URL or path even you can put in a video file or a image path.
    """
    def __init__(self, id, source):
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

    def take_photos(self, num_photos=15, save_dir="images", prefix_name="photo", rotate=0, resize=False, resize_dim=(640,480)):
        """ A simple way to take photos in console and save in a folder.

        Args:
            num_photos: Number of photos to take, 15 by default.
            save_dir: Directory name where the photos will be saved.
            prefix_name: A prefix for the names of the photos.
            rotate: rotate input image around his center, 0 grades by default.
            resize: To resize input image.
            resize_dim: resize input image dimensions (width, height), its default is (640, 480).

        Raises:
            OSError: if folder exist.
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


        if self.type_source == 'other' or self.type_source == 'webcam':
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
                    if rotate != 0:
                        transf_frame = Frame(frame)
                        frame = Rotate(transf_frame).apply(90)
                        frame = cv.flip(frame, 0)
                    if resize:
                        transf_frame = Frame(frame)
                        frame = Resize(transf_frame).apply(resize_dim[0], resize_dim[1])
                    cv.namedWindow("streaming: {}".format(self.id), cv.WINDOW_NORMAL)
                    cv.imshow("streaming: {}".format(self.id), frame)
                elif self.type_source == 'other' or 'webcam':
                    _, frame = cap.read()
                    if rotate != 0:
                        transf_frame = Frame(frame)
                        frame = Rotate(transf_frame).apply(90)
                        frame = cv.flip(frame, -1)
                    if resize:
                        transf_frame = Frame(frame)
                        frame = Resize(transf_frame).apply(resize_dim[0], resize_dim[1])
                    cv.imshow("webcam: {}".format(self.id), frame)
                input_key = cv.waitKey(1)
                
                if input_key == 32:
                    #space pressed
                    cv.imwrite('{}_{}.png'.format(prefix_name, i + 1), frame)
                    print("saved as {}_{}.png".format(prefix_name, i + 1))
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

    def calibrate(self, images_path='', pattern_type='chessboard', pattern_size=(8,5), export_file=True, show=True, fish_eye=True):
        """ Compute camera parameters.

        Args:
            images_path: folder where is saved calibration pattern photos.
            pattern_type: It can be "circles" pattern or "chessboard" pattern (default).
            pattern_size: If pattern_type is "chessboard"  this the Number of inner corners per a chessboard row and column. But If pattern_type is "circles" this will be the number of circles per row and column. 
            export_file: To export camera parameters on xml file.
            show: if is true it show corners or centers found by calibration algorithm  at each iteration.
            fish_eye: A boolean if is true it will calibrate with cv2.fisheye.calibrate if no it'll use normal calibration, fish eye is recomended when cameras has an field of view > 160.

        Returns: 
            RMS calibration's error.

        Raises:
            OSError: If didn't find photos on images_path folder.
            ValueError: If pattern_type is different of 'chessboard' or 'circles' or when
            fish_eye isn't a boolean.
        """
        valid_patterns = ["chessboard", "circles"]
        if pattern_type not in valid_patterns:
            raise ValueError("pattern_type only can be 'chessboard' or 'circles'") 
        if type(fish_eye) != bool:
            raise ValueError("fish_eye has to be a boolean") 
        # adjust image path
        images_path, _ = os.path.splitext(images_path)
        # available formats
        formats = ['*.svg', '*.png', '*.jpg', '*.bmp', '*.jpeg', '*.raw']
        # get files names
        files_grabbed = [glob.glob((images_path + '/' + e)  if len(images_path) > 0 else e) for e in formats]
        # flatting files list
        images = [item for sublist in files_grabbed for item in sublist]
        
        # termination criteria
        if fish_eye == False:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        else:
            criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        if fish_eye == False:
            objp = np.zeros((pattern_size[1]*pattern_size[0],3), np.float32)
            objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        else:
            objp = np.zeros((pattern_size[1]*pattern_size[0], 1, 3), np.float32)
            objp[:,0, :2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
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
                if pattern_type == 'chessboard' and fish_eye == False:
                    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
                elif pattern_type == 'circles' and fish_eye == False:
                    ret, corners = cv.findCirclesGrid(gray, pattern_size, None)
                elif pattern_type == 'chessboard' and fish_eye == True:
                    ret, corners = cv.findChessboardCorners(gray, pattern_size, flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
                elif pattern_type == 'circles' and fish_eye == True:
                    ret, corners = cv.findCirclesGrid(gray, pattern_size, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)
                    if fish_eye == False:   
                        corners = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    else:
                        corners = cv.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
                    imgpoints.append(corners)

                if show:
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, pattern_size, corners, ret)
                    cv.namedWindow(fname, cv.WINDOW_NORMAL)
                    cv.imshow(fname, img)
                    cv.waitKey(3000)
                    cv.destroyAllWindows()

            if fish_eye:
                flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
                k = np.zeros((3, 3))
                d = np.zeros((4, 1))
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]

            if fish_eye == False:
                error, self.matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                height, width = img.shape[:2]
                self.matrix, self.roi = cv.getOptimalNewCameraMatrix(self.matrix, self.dist_coeffs, (width, height), 1, (width, height))
               
            else:
                error, self.matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], k, d, rvecs, tvecs, flags, (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            print("camera error {}".format(error))
            # export an xml with camera parameters
            if export_file:
                print("Saving parameters!")
                cv_file = cv.FileStorage('{}_parameters.xml'.format(self.id), cv.FILE_STORAGE_WRITE)
                cv_file.write('camera_mtx', self.matrix)
                cv_file.write('distortion_coeff', self.dist_coeffs)
                for i, rvec in enumerate(self.rvecs):
                    cv_file.write('rotation_mtx_col_{}'.format(i), rvec) 
                num_rvecs = i + 1
                cv_file.write('num_rvecs', num_rvecs)
                for i, tvec in enumerate(self.tvecs):
                    cv_file.write('translation_mtx_col_{}'.format(i), tvec)
                num_tvecs = i + 1
                cv_file.write('num_tvecs', num_tvecs)
                if hasattr(self, "roi"):
                    cv_file.write('ROI', self.roi)
                cv_file.release()
                print("Saved as {}_parameters.xml".format(self.id))
        except OSError:
            print("Could not find any image in {}".format(images_path))

        return error

    def get_parameters(self, path):
        """ Loads camera parameters from a xml file
        Args: 
            path: a path where is saved xml file. (Note: if you don't have any parameters you can use calibrate method first and then pass true in export file argument)
        Raises:
            OSError: when path is wrong.
         """

        # file storage read
        cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
        if not cv_file.isOpened():
            raise OSError("Looks like {} doesn't exist!".format(path))
        self.matrix = cv_file.getNode('camera_mtx').mat()
        self.dist_coeffs = cv_file.getNode('distortion_coeff').mat()
        num_rvecs = int(cv_file.getNode('num_rvecs').real())
        self.rvecs = [cv_file.getNode('rotation_mtx_col_{}'.format(i)).mat() for i in range(num_rvecs)]
        num_tvecs = int(cv_file.getNode('num_tvecs').real())
        self.tvecs = [cv_file.getNode('translation_mtx_col_{}'.format(i)).mat() for i in range(num_tvecs)]
        if hasattr(self, "roi"):
            self.roi = tuple(cv_file.getNode('ROI').mat().astype(int).reshape(1, -1)[0])

        cv_file.release()
