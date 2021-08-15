import numpy as np

from image_process.frame_decorator import FrameDecorator
from skimage import exposure

class RandomExposure(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, percent=30):
        """Apply random exposure adjustment on an image"""
        while True:
            random = np.random.randint(0, 100)
            if random < percent:
                image = exposure.adjust_gamma(self._frame.apply(), gamma=0.4, gain=0.9)
                # another exposure algo
                # image = exposure.adjust_log(self._frame.apply())
                break
        return image