from __future__ import annotations
from abc import ABC, abstractmethod


from py2vision.stereo.match_method import Matcher

import cv2 as cv

class StereoSystemBuilder(ABC):
    """
    The Builder interface specifies methods for stereo contoller
    """

    @property
    @abstractmethod
    def get_product(self):
        pass

    @abstractmethod
    def pre_process(self):
        pass

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def post_process(self):
        pass
    
    @abstractmethod
    def estimate_depth_map(self):
        pass

    @abstractmethod
    def estimate_3D_points(self):
        pass


class StereoController:
    """
    The StereoController is only responsible for executing the stereo steps in a particular sequence. It is helpful when producing products according to a specific order or configuration.
    """

    def __init__(self) -> None:
        self._stereo_builder = None

    @property
    def stereo_builder(self) -> StereoSystemBuilder:
        return self._stereo_builder

    @stereo_builder.setter
    def stereo_builder(self, builder: StereoSystemBuilder) -> None:
        """
        To change stereo builder instance used by this class
        """
        self._stereo_builder = builder

    def pre_process_step(self, frameL, frameR, downsample=2):
        """ First, it transform from BGR to gray, next apply rectification and finally apply pyramid subsampling.

        Args: 
            frameL: it's the left frame
            frameR: it's the right frame
            downsample: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor just can be 2, 4, 8, 16, 32, 64. If downsample factor is 1 or None or False won't apply downsampling.

        Returns:
            Both frames to apply stereo correspondence or apply more processing to the images.
        """
        left_for_matcher, right_for_matcher = self.stereo_builder.pre_process(frameL, frameR, downsample)
        return left_for_matcher, right_for_matcher
    
    def get_epilines(self, frameL, frameR):
        """ Draw epilines to both frames.

        Args: 
            frameL (arr): it's the left frame.
            frameR (arr): it's the right frame.

        Returns:
            Two elements, the left and right frame with epilines.
        """
        left_epilines, right_epilines = self.stereo_builder.find_epilines(frameL, frameR)
        return left_epilines, right_epilines
        
    def compute_disparity(self, frameL, frameR, matcher: Matcher, downsample=2, lmbda=8000, sigma=1.5, post_process=True, metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, and finally execute the post process with a wls filter to improve the final result.

        Args: 
            frameL: it's the left frame.
            frameR: it's the right frame.
            matcher: A Matcher instance.
            downsample: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor just can be 2, 4, 8, 16, 32, 64. If downsample factor is 1 or None or False won't apply downsampling.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step
            post_process: if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics: if is true print by console the time of execution of correspondence and post process steps.

        Returns:
            Two elements, disparity map and correspondence method used.
        """
        left_for_matcher, right_for_matcher = self.stereo_builder.pre_process(frameL, frameR, downsample)
        left_disp, right_disp, left_matcher = self.stereo_builder.match(left_for_matcher, right_for_matcher, matcher, metrics=metrics)
        if post_process:
            disparity = self.stereo_builder.post_process(left_for_matcher, left_disp, right_disp, left_matcher, lmbda=lmbda, sigma=sigma, metrics=metrics)
        else:
            disparity = left_disp
        # recover original size
        if downsample in [1, None, False]:
                n_upsamples = 0
        else:
            n_upsamples = [2**p for p in range(1, 7)].index(downsample)
            n_upsamples += 1
        if n_upsamples > 0:
            for i in range(n_upsamples):
                disparity = cv.pyrUp(disparity)

        return disparity, left_matcher

    def compute_disparity_color_map(self, frameL, frameR, matcher: Matcher, downsample=2, lmbda=8000, sigma=1.5, post_process=True, metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, then execute the post process with a wls filter to improve the final result, and finally compute disparity map with color.

        Args: 
            frameL: it's the left frame.
            frameR: it's the right frame.
            matcher: A Matcher instance.
            downsample: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor just can be 2, 4, 8, 16, 32, 64. If downsample factor is 1 or None or False won't apply downsampling.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step.
            post_process: if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics: if is true print by console the time of execution of correspondence and post process steps.

        Returns:
            Two elements, disparity map (with 3 colors channels) and correspondence method used
        """
        disparity, matcher = self.compute_disparity(frameL, frameR, matcher, downsample, float(lmbda), sigma, post_process, metrics)
        disparity_colormap = self.stereo_builder.estimate_disparity_colormap(disparity)
        return disparity_colormap, matcher


    def compute_3D_map(self, frameL, frameR, Q, matcher: Matcher, downsample=2, lmbda=8000, sigma=1.5, post_process=True,  metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, then execute the post process with a wls filter to improve the final result, and finally compute depth map.

        Args: 
            frameL: it's the left frame.
            frameR: it's the right frame.
            Q: a 4x4 array with the following structure, [[1 0   0          -cx     ][0 1   0          -cy     ][0 0   0           f      ][0 0 -1/Tx (cx - cx')/Tx ]], cx: is the principal point x in left image, cx': is the principal point x in right image, cy: is the principal point y in left image, f: is the focal lenth in left image, Tx: The x coordinate in Translation matrix.  
            matcher: A Matcher instance.
            downsample: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor just can be 2, 4, 8, 16, 32, 64. If downsample factor is 1 or None or False won't apply downsampling.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step
            post_process: if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics: if is true print by console the time of execution of correspondence, post process and depth map step.

        Returns:
            reprojected points image, a depth map in RGB and correspondence method used
        """
        disparity, matcher = self.compute_disparity(frameL, frameR, matcher, downsample, float(lmbda), sigma, post_process, metrics)
        return self.stereo_builder.estimate_depth_map(disparity, Q, frameL, metrics=metrics), matcher

    def compute_3D_points(self, frameL, frameR, points, Q, matcher: Matcher, downsample=2, lmbda=8000, sigma=1.5, post_process=True,  metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, then execute the post process with a wls filter to improve the final result, and finally compute 3D points for an input array that can be [[x1, y1], [x2, y2], [x3, y3], ... [xn, yn]].

        Args: 
            frameL: it's the left frame.
            frameR: it's the right frame.
            points: contains the (x, y) coordinates in image plane to convert to (X, Y, Z)
            Q: a 4x4 array with the following structure, [[1 0   0          -cx     ][0 1   0          -cy     ][0 0   0           f      ][0 0 -1/Tx (cx - cx')/Tx ]], cx: is the principal point x in left image, cx': is the principal point x in right image, cy: is the principal point y in left image, f: is the focal lenth in left image, Tx: The x coordinate in Translation matrix.  
            matcher: A Matcher instance.
            downsample: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor just can be 2, 4, 8, 16, 32, 64. If downsample factor is 1 or None or False won't apply downsampling.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step
            post_process: if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics: if is true print by console the time of execution of correspondence, post process and depth map step.
            
        Returns:
            Two elements, an array of points in 3D homogeneous coordinates (X, Y, Z, W) and correspondence method used
        """
        disparity, matcher = self.compute_disparity(frameL, frameR, matcher, downsample, float(lmbda), sigma, post_process, metrics)
        return self.stereo_builder.estimate_3D_points(points, disparity, Q), matcher