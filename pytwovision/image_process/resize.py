import cv2 as cv

from pytwovision.image_process.frame_decorator import FrameDecorator

class Resize(FrameDecorator):
    def apply(self, width, height):
        """Apply resizing on an image.
        
        Args:
             width: new width to the image.
             height: new height to the image.
        
        Returns:
            an image resized.
        """
        return cv.resize(self._frame.apply(), (width, height), interpolation= cv.INTER_LINEAR_EXACT)