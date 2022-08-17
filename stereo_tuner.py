from __future__ import annotations
import cv2 as cv
import json
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from pytwovision.stereo.standard_stereo import StandardStereoBuilder
from pytwovision.input_output.camera import Camera
from pytwovision.stereo.standard_stereo import StandardStereo
from pytwovision.stereo.match_method import Matcher, StereoSGBM

left_camera = Camera("left_camera", "imx_219_A")
right_camera = Camera("right_camera", "imx_219_B")
stereo_pair_fisheye = StandardStereo(left_camera, right_camera)
stereo_maps_path = "stereoMap"
stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True, export_file_name=stereo_maps_path)
stereo_pair_fisheye.print_parameters()
builder = StandardStereoBuilder(left_camera, right_camera, stereo_maps_path + ".xml")
left = cv.imread("tests/assets/photo/left/left_indoor_photo_5.png")
right = cv.imread("tests/assets/photo/right/right_indoor_photo_5.png")

global left_for_matcher, right_for_matcher, post_process 
downsample = None
post_process = True
min_disp = -16
num_disp = 32
window_size = 3
p1 = 125
p2 = 1376
pre_filter_cap = 57
speckle_window_size = 114
speckle_range = 9
uniqueness_ratio = 3
disp_12_max_diff = -38
lmbda = 19132
sigma = 1.046

# rectified images
left_for_matcher, right_for_matcher = builder.pre_process(left, right, downsample)

fig = plt.subplots(1,2)
if post_process:
    plt.subplots_adjust(left=0.15, bottom=0.55)
else:
    plt.subplots_adjust(left=0.15, bottom=0.5)
plt.subplot(1,2,1)
dmObject = plt.imshow(left_for_matcher, 'gray')

def get_stereo_map():
  sgbm = StereoSGBM(min_disp=min_disp, max_disp=num_disp, window_size=window_size, p1=p1, p2=p2, pre_filter_cap=pre_filter_cap, speckle_window_size=speckle_window_size, speckle_range=speckle_range, uniqueness_ratio=uniqueness_ratio, disp_12_max_diff=disp_12_max_diff)
  matcher = Matcher(sgbm)
  left_disp, right_disp, left_matcher = builder.match(left_for_matcher, right_for_matcher, matcher, metrics=False)
  # To postprocess
  if post_process:
    left_disp = builder.post_process(left_for_matcher, left_disp, right_disp, left_matcher, lmbda=lmbda, sigma=sigma, metrics=False)
  if downsample in [1, None, False]:
    n_upsamples = 0
  else:
    n_upsamples = [2**p for p in range(1, 7)].index(downsample)
    n_upsamples += 1
  if n_upsamples > 0:
    for i in range(n_upsamples):
        left_disp = cv.pyrUp(left_disp)
  #left_disp = builder.estimate_disparity_colormap(left_disp)
  return left_disp

def load_settings(event):
  global min_disp, num_disp, window_size, p1, p2, pre_filter_cap, speckle_range, uniqueness_ratio, disp_12_max_diff, speckle_window_size, loading_settings, lmbda, sigma
  loading_settings = 1
  file_name = 'SGBM_disparity_parameters_set.txt'
  print('Loading parameters from file...')
  button_load.label.set_text ("Loading...")
  f=open(file_name, 'r')
  data = json.load(f)
  slider_min_disparity.set_val(data['minDisparity'])
  slider_num_disparity.set_val(data['numberOfDisparities'])
  slider_block_size.set_val(data['windowSize'])
  slider_p1.set_val(data['p1'])
  slider_p2.set_val(data['p2'])
  slider_pre_filter_cap.set_val(data['preFilterCap'])
  slider_speckle_window_size.set_val(data['speckleWindowSize'])
  slider_speckle_range.set_val(data['speckleRange'])
  slider_uniqueness_ratio.set_val(data['uniquenessRatio'])
  slider_disp_12_max_diff.set_val(data['disp12MaxDiff'])
  if post_process:
    slider_lambda.set_val(data['lambda'])
    slider_sigma.set_val(data['sigma'])
  f.close()
  button_load.label.set_text ("Load settings")
  print ('Parameters loaded from file '+ file_name)
  print ('Redrawing depth map with loaded parameters...')
  loading_settings = 0
  update(0)
  print ('Done!')

def save_settings(event):
  button_save.label.set_text ("Saving...")
  print('Saving to file...') 
  if post_process:
    result = json.dumps({'windowSize' : window_size, 'preFilterCap' : pre_filter_cap, \
        'minDisparity' : min_disp, 'numberOfDisparities' : num_disp, 'p1' : p1, \
        'p2' : p2, 'uniquenessRatio': uniqueness_ratio, 'disp12MaxDiff' : disp_12_max_diff, \
        'speckleRange': speckle_range, 'speckleWindowSize': speckle_window_size, 'lambda': lmbda, 'sigma': sigma},\
        sort_keys=True, indent=4, separators=(',',':'))
  else:
    result = json.dumps({'windowSize' : window_size, 'preFilterCap' : pre_filter_cap, \
        'minDisparity' : min_disp, 'numberOfDisparities' : num_disp, 'p1' : p1, \
        'p2' : p2, 'uniquenessRatio': uniqueness_ratio, 'disp12MaxDiff' : disp_12_max_diff, \
        'speckleRange': speckle_range, 'speckleWindowSize': speckle_window_size},\
        sort_keys=True, indent=4, separators=(',',':'))
  file_name = 'SGBM_disparity_parameters_set.txt'
  f = open (str(file_name), 'w') 
  f.write(result)
  f.close()
  button_save.label.set_text ("Save to file")
  print ('Settings saved to file ' + file_name)
    
axcolor = 'lightgoldenrodyellow'
if post_process:
    save_axe = plt.axes([0.3, 0.46, 0.15, 0.04]) #stepX stepY width height
    button_save = Button(save_axe, 'Save settings', color=axcolor, hovercolor='0.975')
    button_save.on_clicked(save_settings)
    load_axe = plt.axes([0.5, 0.46, 0.15, 0.04]) #stepX stepY width height
    button_load = Button(load_axe, 'Load settings', color=axcolor, hovercolor='0.975')
    button_load.on_clicked(load_settings)
else:
    save_axe = plt.axes([0.3, 0.38, 0.15, 0.04]) #stepX stepY width height
    button_save = Button(save_axe, 'Save settings', color=axcolor, hovercolor='0.975')
    button_save.on_clicked(save_settings)
    load_axe = plt.axes([0.5, 0.38, 0.15, 0.04]) #stepX stepY width height
    button_load = Button(load_axe, 'Load settings', color=axcolor, hovercolor='0.975')
    button_load.on_clicked(load_settings)
disparity = get_stereo_map()
plt.subplot(1,2,2)
dmObject = plt.imshow(disparity, "gray")

min_disparity_axe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
num_disparity_axe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
block_size_axe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
p1_axe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
p2_axe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
pre_filter_cap_axe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
speckle_window_size_axe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
speckle_range_axe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
uniqueness_ratio_axe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
disp_12_max_diff_axe = plt.axes([0.15, 0.37, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
if post_process:
  lambda_axe = plt.axes([0.15, 0.41, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
  sigma_axe = plt.axes([0.15, 0.45, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height

slider_min_disparity = Slider(min_disparity_axe, "min disparity", -256, 256, valinit=min_disp)
slider_num_disparity = Slider(num_disparity_axe, "num disparities", 0, 512, valinit=num_disp)
slider_block_size = Slider(block_size_axe, "block size", 2, 51, valinit=window_size)
slider_p1 = Slider(p1_axe, "P1", 0, 3000, valinit=p1)
slider_p2 = Slider(p2_axe, "P2", 0, 8000, valinit=p2)
slider_pre_filter_cap = Slider(pre_filter_cap_axe, "Pre filter cap", 0, 63, valinit=pre_filter_cap)
slider_speckle_window_size = Slider(speckle_window_size_axe, "Speckle window size", 0, 500, valinit=speckle_window_size)
slider_speckle_range = Slider(speckle_range_axe, "Speckle range", 0, 50, valinit=speckle_range)
slider_uniqueness_ratio = Slider(uniqueness_ratio_axe, "Uniqueness ratio", 1, 20, valinit=uniqueness_ratio)
slider_disp_12_max_diff = Slider(disp_12_max_diff_axe, "Disp 12 max diff", -64, 128, valinit=disp_12_max_diff)
if post_process:
  slider_lambda = Slider(lambda_axe, "Lambda", 0, 26000, valinit=lmbda)
  slider_sigma = Slider(sigma_axe, "sigma", 0, 5, valinit=sigma)

def update(val):
  global min_disp, num_disp, window_size, p1, p2, pre_filter_cap, speckle_range, uniqueness_ratio, disp_12_max_diff, speckle_window_size,lmbda, sigma
  #update sliders
  min_disp = int(slider_min_disparity.val)
  num_disp = int(slider_num_disparity.val/16)*16
  if slider_block_size.val % 2 == 0:
    slider_block_size.val += 1
  window_size = int(slider_block_size.val)
  p1 = int(slider_p1.val)
  p2 = int(slider_p2.val)
  pre_filter_cap = int(slider_pre_filter_cap.val)
  speckle_window_size = int(slider_speckle_window_size.val)
  speckle_range = int(slider_speckle_range.val)
  uniqueness_ratio = int(slider_uniqueness_ratio.val)
  disp_12_max_diff = int(slider_disp_12_max_diff.val)
  if post_process:
    lmbda = slider_lambda.val
    sigma = slider_sigma.val
  assert ValueError
  if (loading_settings==0):
    print ('Rebuilding depth map')
    disparity = get_stereo_map()
    dmObject.set_data(disparity)
    print ('Redraw depth map')
    plt.draw()

slider_min_disparity.on_changed(update)
slider_num_disparity.on_changed(update)
slider_block_size.on_changed(update)
slider_p1.on_changed(update)
slider_p2.on_changed(update)
slider_pre_filter_cap.on_changed(update)
slider_speckle_window_size.on_changed(update)
slider_speckle_range.on_changed(update)
slider_uniqueness_ratio.on_changed(update)
slider_disp_12_max_diff.on_changed(update)

if post_process:
  slider_lambda.on_changed(update)
  slider_sigma.on_changed(update)

print('Show interface to user')
plt.show()