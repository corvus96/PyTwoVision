# A package to do stereo vision using object detection

py2vision will allow you to determine the
homogeneous coordinates of a detected object with a neural network, in conjunction with stereo algorithms that allow you to find the disparity maps, to finally apply segmentation within each detection window and it is these segmented pixels that are used to find the real position of the object in 3D space.
These segmented pixels are used to find the actual position of the object in 3D space.

## **Installation**
Open a console and write this: 

`pip install py2vision`

**Note**: If you already have the opencv package installed, you have to uninstall it with pip uninstall opencv-python and install opencv-contrib-python version 4.6.0.66.

## **Application areas**
- Robotics
- Computer Vision
- Web apps with Object detection
- Surveillance

If you want to learn how to use it I recommend see inside tutorials folder because it have some tools and notebooks that can help you to understand how it's works.

Or you can see our documentation website here: https://corvus96.github.io/PyTwoVision/html/index.html
