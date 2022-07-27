import numpy as np

from pytwovision.image_process.frame_decorator import FrameDecorator
from skimage import exposure

class RandomExposure(FrameDecorator):
    def apply(self, percent=30):
        """Apply random exposure adjustment on an image
        Arguments:
            percent: a float between [0, 100]
        Returns:
            An image array modified
        """
        while True:
            random = np.random.randint(0, 100)
            if random < percent:
                image = exposure.adjust_gamma(self._frame.apply(), gamma=0.4, gain=0.9)
                # another exposure algo
                # image = exposure.adjust_log(self._frame.apply())
                break
        return image