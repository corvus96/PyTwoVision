=======
Compute 
=======
This module was created in order to house the calculations that may be needed by the stereo mechanisms and those required by the neural networks. 

Calculate the reprojection error
--------------------------
It can be useful when you need to obtain the calibration of the error of the cameras, the calibration is when you find the intrinsic and extrinsic parameters of a camera, such as: distortions, center of projections, focal lengths and so on.

    .. automodule:: py2vision.compute.error_compute
        :members:

Yolo V3 Computations
---------------------
The class YoloV3Calculus is responsible for adding the prediction layer in a YOLO version 3 network, converting the matrices containing the bounding box coordinates from a format ($xmin$, $ymin$, $xmax$, $ymax$) to a format ($cx$, $cy$, $w$, $h$) where $cx$ and $cy$ are the coordinates of the center point of the bounding box and the variables $w$ and $h$ are the width and height of the box respectively, such conversion can be done in both directions.

It is also in charge of calculating the IoU value and applying the **Non maximum suppression** algorithm to filter the bounding boxes that do not meet a certain threshold, in order to obtain predictions with a lower amount of noise and calculate the network error.

    .. automodule:: py2vision.compute.yolov3_calculus
        :members:


