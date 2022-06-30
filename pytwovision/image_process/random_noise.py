import numpy as np

from pytwovision.image_process.frame_decorator import FrameDecorator
from skimage.util import random_noise

class RandomNoise(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, percent=30):
        """Apply random noise on an image"""
        while True:
            random = np.random.randint(0, 100)
            if random < percent:
                image = random_noise(self._frame.apply())
                break
        return image