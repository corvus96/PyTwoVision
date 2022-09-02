import cv2 as cv
from py2vision.image_process.frame_decorator import FrameDecorator
class Rotate(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, angle):
        """Apply rotation on an image.

        Args: 
            angle: an integer that represents rotation angle.

        Returns: 
            an image rotated.
        """
        # dividing height and width by 2 to get the center of the image
        height, width = self._frame.apply().shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width/2, height/2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
        return cv.warpAffine(src=self._frame.apply(), M=rotate_matrix, dsize=(width, height))