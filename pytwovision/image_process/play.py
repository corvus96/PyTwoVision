from random_exposure import RandomExposure
from frame_decorator import Frame
from random_noise import RandomNoise
from random_intensity  import RandomIntensityRescale
import cv2 as cv

im = cv.imread('download.jpeg')
cv.imshow('original', im)
fr = Frame(im)
random_exposure_im = RandomExposure(fr).apply()
cv.imshow('random exposure',random_exposure_im)
random_noise_im = RandomNoise(fr).apply()
cv.imshow('random noise',random_noise_im)
random_intsensity_im = RandomIntensityRescale(fr).apply()
cv.imshow('random intensity',random_noise_im)
fr2 = Frame(random_intsensity_im)
r4 = RandomExposure(fr2).apply()
cv.imshow("test", r4)
cv.waitKey(0) 

