import cv2 as cv

from __future__ import annotations
from abc import ABC, abstractmethod


class Matcher():
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: MatcherStrategy) -> None:
        """
        The matcher accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

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


"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class StereoSGBM(MatcherStrategy):
    def __init__(self, min_disp=0, max_disp=160, window_size=3, p1=24*3*3, p2=96*3*3, pre_filter_cap=63, mode=cv.StereoSGBM_MODE_SGBM_3WAY, speckle_window_size=1100, speckle_range=1, uniqueness_ratio=5, disp_12_max_diff=-1):
        try:
            if max_disp <= 0 or max_disp % 16 != 0:
                raise ValueError
        except ValueError:
            print("Incorrect max_disparity value: it should be positive and divisible by 16")
            exit()
        try:
            if window_size <= 0 or window_size % 2 != 1:
                raise ValueError
        except ValueError:
            print("Incorrect window_size value: it should be positive and odd")
            exit()
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
        return cv.StereoSGBM_create(self.min_disp, int(self.max_disp), self.window_size, self.p1, self.p2, self.disp_12_max_diff, self.pre_filter_cap, self.uniqueness_ratio, self.speckle_window_size, self.speckle_range, self.mode)