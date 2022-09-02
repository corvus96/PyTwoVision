from __future__ import annotations
from abc import ABC, abstractmethod

import cv2 as cv


class Matcher():
    """
    The Context defines the interface of interest to clients.
    The matcher accepts a strategy through the constructor, but also provides a setter to change it at runtime.
    """

    def __init__(self, strategy: MatcherStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> MatcherStrategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: MatcherStrategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def match(self):
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        return self._strategy.match()



class MatcherStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def match(self):
        pass

class StereoSGBM(MatcherStrategy):
    """ To create an instance of stereo SGBM algorithm.

    Args:
        min_disp: Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        max_disp: Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        window_size: Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        p1: The first parameter controlling the disparity smoothness. 
        p2: The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*SADWindowSize*SADWindowSize and 32*number_of_image_channels*SADWindowSize*SADWindowSize, respectively).
        pre_filter_cap: Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
        mode: Set it to StereoSGBM_MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .
        speckle_window_size: Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckle_range: Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        uniqueness_ratio: Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        disp_12_max_diff: Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    """
    def __init__(self, min_disp=0, max_disp=160, window_size=3, p1=24*3*3, p2=96*3*3, pre_filter_cap=63, mode=cv.StereoSGBM_MODE_HH, speckle_window_size=1100, speckle_range=1, uniqueness_ratio=5, disp_12_max_diff=-1):
        try:
            if max_disp <= 0 or max_disp % 16 != 0:
                raise ValueError
        except ValueError:
            print("Incorrect max_disparity value: it should be positive and divisible by 16")
        try:
            if window_size <= 0 or window_size % 2 != 1:
                raise ValueError
        except ValueError:
            print("Incorrect window_size value: it should be positive and odd")
        max_disp /= 2
        if(max_disp % 16 != 0):
            max_disp += 16-(max_disp % 16)

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.window_size = window_size
        self.p1 = p1
        self.p2 = p2
        self.pre_filter_cap = pre_filter_cap
        self.mode = mode
        self.speckle_window_size = speckle_window_size
        self.speckle_range = speckle_range
        self.uniqueness_ratio = uniqueness_ratio
        self.disp_12_max_diff = disp_12_max_diff

    def match(self):
        """ Return stereo sgbm instance """
        return cv.StereoSGBM_create(self.min_disp, int(self.max_disp), self.window_size, self.p1, self.p2, self.disp_12_max_diff, self.pre_filter_cap, self.uniqueness_ratio, self.speckle_window_size, self.speckle_range, self.mode)