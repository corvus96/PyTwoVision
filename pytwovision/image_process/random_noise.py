import numpy as np

from pytwovision.image_process.frame_decorator import FrameDecorator
from skimage.util import random_noise

class RandomNoise(FrameDecorator):
    def apply(self, percent=30):
        """Apply random noise on an image
        Arguments:
            percent: a float between [0, 100]
        Returns:
            An image array modified
        """
        while True:
            random = np.random.randint(0, 100)
            if random < percent:
                image = random_noise(self._frame.apply())
                break
        return image