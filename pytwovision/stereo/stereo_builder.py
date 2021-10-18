from __future__ import annotations
from abc import ABC, abstractmethod

from numpy.lib.function_base import disp

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
    The StereoController is only responsible for executing the stereo steps in a
    particular sequence. It is helpful when producing products according to a
    specific order or configuration.
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

    """
    The StereoController can construct several product variations using the same
    building steps.
    """
    def pre_process_step(self, frameL, frameR, downsample=True):
        """ First, it transform from BGR to gray, next apply rectification and finally apply pyramid subsampling.
        Arguments: 
            frameL (arr): it's the left frame
            frameR (arr): it's the right frame
            downsample (bool): if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor is 2.
        Returns:
            Both frames to apply stereo correspondence or apply more processing to the images.
        """
        left_for_matcher, right_for_matcher = self.stereo_builder.pre_process(frameL, frameR, downsample)
        return left_for_matcher, right_for_matcher
    
    def get_epilines(self, frameL, frameR):
        """ Draw epilines to both frames
        Arguments: 
            frameL (arr): it's the left frame
            frameR (arr): it's the right frame
        Returns:
            Two elements, the left and right frame with epilines
        """
        left_epilines, right_epilines = self.stereo_builder.find_epilines(frameL, frameR)
        return left_epilines, right_epilines
        
    def compute_disparity(self, frameL, frameR, downsample=True, post_process=True, metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, and finally execute the post process with a wls filter to improve the final result
         Arguments: 
            frameL (arr): it's the left frame
            frameR (arr): it's the right frame
            downsample (bool): if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor is 2.
            post_process (bool): if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics (bool): if is true print by console the time of execution of correspondence and post process steps.
        Returns:
            Two elements, disparity map and correspondence method used
        """
        left_for_matcher, right_for_matcher = self.stereo_builder.pre_process(frameL, frameR, downsample)
        left_disp, right_disp, left_matcher = self.stereo_builder.match(left_for_matcher, right_for_matcher, metrics=metrics)
        if post_process:
            disparity = self.stereo_builder.post_process(frameL, left_disp, right_disp, left_matcher, metrics=metrics)
        else:
            disparity = left_disp
        return disparity, left_matcher

    def compute_disparity_color_map(self, frameL, frameR, downsample=True, post_process=True, metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, then execute the post process with a wls filter to improve the final result, and finally compute disparity map with color.
         Arguments: 
            frameL (arr): it's the left frame
            frameR (arr): it's the right frame
            downsample (bool): if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor is 2.
            post_process (bool): if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics (bool): if is true print by console the time of execution of correspondence and post process steps.
        Returns:
            Two elements, disparity map (with 3 colors channels) and correspondence method used
        """
        disparity, matcher = self.compute_disparity(frameL, frameR, downsample, post_process, metrics)
        disparity_colormap = self.stereo_builder.estimate_disparity_colormap(disparity)
        return disparity_colormap, matcher


    def compute_3D_map(self, frameL, frameR, Q, downsample=True, post_process=True,  metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, then execute the post process with a wls filter to improve the final result, and finally compute depth map.
         Arguments: 
            frameL (arr): it's the left frame
            frameR (arr): it's the right frame
            Q (arr): a 4x4 array with the following structure, 
                    [[1 0   0          -cx     ]
                     [0 1   0          -cy     ]
                     [0 0   0           f      ]
                     [0 0 -1/Tx (cx - cx')/Tx ]]
                    cx: is the principal point x in left image
                    cx': is the principal point x in right image
                    cy: is the principal point y in left image
                    f: is the focal lenth in left image
                    Tx: The x coordinate in Translation matrix   
            downsample (bool): if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor is 2.
            post_process (bool): if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics (bool): if is true print by console the time of execution of correspondence, post process and depth map step.
        Returns:
            Two elements, depth map and correspondence method used
        """
        left_for_matcher, right_for_matcher = self.stereo_builder.pre_process(frameL, frameR, downsample)
        left_disp, right_disp, left_matcher = self.stereo_builder.match(left_for_matcher, right_for_matcher, metrics=metrics)
        if post_process:
            disparity = self.stereo_builder.post_process(frameL, left_disp, right_disp, left_matcher, metrics=metrics)
        else:
            disparity = left_disp
        return self.stereo_builder.estimate_depth_map(disparity, Q, left_for_matcher, metrics=metrics), left_matcher

    def compute_3D_points(self, frameL, frameR, points, Q, downsample=True, post_process=True,  metrics=True):
        """ Apply the pre process step, next compute left and right disparity maps, then execute the post process with a wls filter to improve the final result, and finally compute 3D points for an input array that can be [[x1, y1], [x2, y2], [x3, y3], ... [xn, yn]].
         Arguments: 
            frameL (arr): it's the left frame
            frameR (arr): it's the right frame
            points (arr): contains the (x, y) coordinates in image plane to convert to (X, Y, Z)
            Q (arr): a 4x4 array with the following structure, 
                    [[1 0   0          -cx     ]
                     [0 1   0          -cy     ]
                     [0 0   0           f      ]
                     [0 0 -1/Tx (cx - cx')/Tx ]]
                    cx: is the principal point x in left image
                    cx': is the principal point x in right image
                    cy: is the principal point y in left image
                    f: is the focal lenth in left image
                    Tx: The x coordinate in Translation matrix   
            downsample (bool): if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor is 2.
            post_process (bool): if is true apply post_process and return improved disparity map, otherwise return left disparity map without post processing.
            metrics (bool): if is true print by console the time of execution of correspondence, post process and depth map step.
        Returns:
            Two elements, an array of points in 3D homogeneous coordinates (X, Y, Z, W) and correspondence method used
        """
        left_for_matcher, right_for_matcher = self.stereo_builder.pre_process(frameL, frameR, downsample)
        left_disp, right_disp, left_matcher = self.stereo_builder.match(left_for_matcher, right_for_matcher, metrics=metrics)
        if post_process:
            disparity = self.stereo_builder.post_process(frameL, left_disp, right_disp, left_matcher, metrics=metrics)
        else:
            disparity = left_disp
        return self.stereo_builder.estimate_3D_points(points, disparity, Q), left_matcher