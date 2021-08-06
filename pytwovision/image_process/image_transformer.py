import numpy as np

from skimage.util import random_noise
from skimage import exposure

class ImageTransformer:
    def __init__(self, input_image):
        self.input = input_image
        self.output = None

    def apply_random_noise(self, image, percent=30):
        """Apply random noise on an image"""
        random = np.random.randint(0, 100)
        if random < percent:
            image = random_noise(image)
        return image


    def apply_random_intensity_rescale(self, image, percent=30):
        """Apply random intensity rescale on an image"""
        random = np.random.randint(0, 100)
        if random < percent:
            v_min, v_max = np.percentile(image, (0.2, 99.8))
            image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        return image


    def apply_random_exposure_adjust(self, image, percent=30):
        """Apply random exposure adjustment on an image"""
        random = np.random.randint(0, 100)
        if random < percent:
            image = exposure.adjust_gamma(image, gamma=0.4, gain=0.9)
            # another exposure algo
            # image = exposure.adjust_log(image)
        return image