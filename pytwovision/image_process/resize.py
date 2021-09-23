import cv2 as cv

from image_process.frame_decorator import FrameDecorator

class Resize(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, width, height):
        """Apply resizing on an image"""
        return cv.resize(self._frame.apply(), (width, height), interpolation= cv.INTER_LINEAR_EXACT)