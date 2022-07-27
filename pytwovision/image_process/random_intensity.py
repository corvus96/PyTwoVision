import numpy as np

from pytwovision.image_process.frame_decorator import FrameDecorator
from skimage import exposure

class RandomIntensityRescale(FrameDecorator):
    def apply(self, percent=30):
        """Apply random intensity rescale on an image
        Arguments:
            percent: a float between [0, 100]
        Returns:
            An image array modified
        """
        while True:
            random = np.random.randint(0, 100)
            if random < percent:
                v_min, v_max = np.percentile(self._frame.apply(), (0.2, 99.8))
                image = exposure.rescale_intensity(self._frame.apply(), in_range=(v_min, v_max))
                break
        return image