from __future__ import annotations
import cv2 as cv
from abc import ABC, abstractmethod
from typing import List

class StereoTrackBars(ABC):
    """
    An interface that declares a set of methods for managing a type of track bar. 
    Track Bar type depends of stereo correspondence method used.
    """

    @abstractmethod
    def attach(self, observer: TrackBarObserver, init_value, max_value) -> None:
        """
        Attach a trackbar to the display.
        """
        pass

    @abstractmethod
    def detach(self, observer: TrackBarObserver) -> None:
        """
        Detach a trackbar to the display.
        """
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Notify all updaters about an event.
        """
        pass
        
class StereoSGBMTrackBars(StereoTrackBars):
    """
    It owns some important state and notifies updaters when the state
    changes.
    """
    _observers: List[TrackBarObserver] = []
    def __init__(self, window_name):
        """Initialize state variables
        Args:
            window_name (str): display name to append trackbars
        """
        _num_disparities = None
        _block_size = None
        _disp_12_max_diff = None
        _min_disparity = None
        _p1 = None
        _p2 = None
        _pre_filter_cap = None
        _speckle_range = None
        _uniqueness_ratio = None
        _speckle_window_size = None
        self.window_name = window_name

    def attach(self, observer: TrackBarObserver, init_value, max_value):
        """ Add a trackbar to update state variable
        Args: 
            observer (object): expect an variable updater.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        self._observers.append(observer)
        observer.create(self, init_value, max_value)
        

    def detach(self, observer: TrackBarObserver) -> None:
        """ Delete trackbar on window
        Args: 
            observer (object): expect an variable updater.
        """
        self._observers.remove(observer)

    """
    The subscription management methods.
    """

    def notify(self, matcher) -> None:
        """
        Trigger an update in each trackbar.
        """
        for observer in self._observers:
            observer.update(self, matcher)
    
    def tune_matcher(self, matcher) -> None:
        """ 
        Execute update atached Trackbars updaters
        """
        self.notify(matcher)

class TrackBarObserver(ABC):
    """
    The Observer interface declares the update method, used by Track bars.
    """

    @abstractmethod
    def create(self, subject: StereoTrackBars, init_value, max_value) -> None:
        """
        Create trackbar.
        """
        pass
    @abstractmethod
    def update(self, subject: StereoTrackBars) -> None:
        """
        Receive update from StereoTrackBars.
        """
        pass

class NumDisparitiesTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=80, max_value=256):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('numDisparities', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._num_disparities = cv.getTrackbarPos('numDisparities', subject.window_name) * 16 
        matcher.setNumDisparities(subject._num_disparities)

class BlockSizeTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=3, max_value=51):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('blockSize', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._block_size = cv.getTrackbarPos('blockSize', subject.window_name) 
        if subject._block_size % 2 == 0:
            subject._block_size += 1
        matcher.setBlockSize(subject._block_size)
        

class MinDisparityTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=0, max_value=128):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('minDisparity', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._min_disparity = cv.getTrackbarPos('minDisparity', subject.window_name) 
        matcher.setMinDisparity(subject._min_disparity)
        

class Disp12MaxDiffTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=0, max_value=512):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('disp12MaxDiff', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._disp_12_max_diff = cv.getTrackbarPos('disp12MaxDiff', subject.window_name) 
        matcher.setDisp12MaxDiff(subject._disp_12_max_diff)
     
class P1TrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=216, max_value=432):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('p1', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._p1 = cv.getTrackbarPos('p1', subject.window_name) 
        matcher.setP1(subject._p1)

class P2TrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=864, max_value=1728):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('p2', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._p2 = cv.getTrackbarPos('p2', subject.window_name) 
        matcher.setP2(subject._p2)


class PreFilterCapTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=1, max_value=63):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('preFilterCap', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._pre_filter_cap = cv.getTrackbarPos('preFilterCap', subject.window_name) 
        matcher.setPreFilterCap(subject._pre_filter_cap)
       
class SpeckleRangeTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=0, max_value=100):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('speckleRange', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._speckle_range = cv.getTrackbarPos('speckleRange', subject.window_name) 
        matcher.setSpeckleRange(subject._speckle_range)

class SpeckleWindowSizeTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=50, max_value=200):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('speckleWindowSize', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._speckle_window_size = cv.getTrackbarPos('speckleWindowSize', subject.window_name) 
        matcher.setSpeckleWindowSize(subject._speckle_window_size)

class UniquenessRatioTrackBarUpdater(TrackBarObserver):
    def create(self, subject: StereoTrackBars, init_value=5, max_value=15):
        """ Add a trackbar to update state variable
        Args: 
            subject (object): subscribe to a stereo track bar.
            init_value (int): initial value in trackbar.
            max_value (int): trackbar range.
        """
        cv.createTrackbar('uniquenessRatio', subject.window_name, init_value, max_value, (lambda a: None))

    def update(self, subject: StereoTrackBars, matcher):
        """ Update trackbar value when the user interacts with UI
        Args: 
            matcher: identify type of stereo correspondence
        """
        subject._uniqueness_ratio = cv.getTrackbarPos('uniquenessRatio', subject.window_name) 
        matcher.setUniquenessRatio(subject._uniqueness_ratio)
        
