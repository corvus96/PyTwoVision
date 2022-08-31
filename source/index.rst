.. Pytwovision documentation master file, created by
   sphinx-quickstart on Mon Jul 11 02:45:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
Welcome to Pytwovision's documentation!
=======================================
Here you can see every detail about pytwovision package which is a tool to do object positioning with stereo vision and object detection, this package can be used in Robotics and Computer Vision.

Installation
------------
Open a console and write this:
:command:`pip install pytwovision`

.. important::

   If you already have the opencv package installed, you have to uninstall it with :command:`pip uninstall opencv-python` and install opencv-contrib-python version 4.6.0.66.

What could you archieve with pytwovision?
-----------------------------------------
#. It will allow you to obtain disparity maps using SGBM algorithm.
#. Get distances in the 3D space.
#. Measure objects dimensions and more.
#. Train your own object detection model and infer with it.
#. Combine detection and stereo vision to get homogeneous coordinates of objects im an scene.

.. figure:: https://github.com/corvus96/PyTwoVision/blob/master/tests/test_dataset/images/2007_000027.jpg?raw=true
   :height: 250
   :width: 250
   :align: center

   Example of a disparity map

.. figure:: https://github.com/corvus96/PyTwoVision/blob/master/tests/test_dataset/images/2007_000027.jpg?raw=true
   :height: 250
   :width: 250
   :align: center
   
   Example of positioning of objects

.. toctree::
   :maxdepth: 2
   :caption: Modules
   :glob:

   module*

Dependencies
---------------
* numpy == 1.21.5
* tensorflow == 2.8.0
* opencv-contrib-python==4.6.0.66
* wget == 3.2
* matplotlib==3.5.1
* pandas
* pyyaml 
* h5py

References
==========

.. [1] C. Wheatstone, «Contributions to the Physiology of Vision.», Proceedings
of the Royal Society of London, vol. 4, 0 1837.

.. [2] M. M. Martín, Técnicas de visión estereoscópica para determinar la estruc-
tura tridimensional de la escena Proyecto, Madrid, 2010.

.. [3] J. Dembys, Y. Gao, A. Shafiekhani y G. Desouza, Object Detection and Pose
Estimation Using CNN in Embedded Hardware for Assistive Technology,
oct. de 2019.

.. [4] H. Königshof, N. O. Salscheider y C. Stiller, «Realtime 3D Object Detection
for Automated Driving Using Stereo Vision and Semantic Information», en
2019 IEEE Intelligent Transportation Systems Conference (ITSC), 2019,
págs. 1405-1410. doi: 10.1109/ITSC.2019.8917330.

.. [5] S. T. Barnard y M. A. Fischler, «Computational Stereo», ACM Computing
Surveys (CSUR), vol. 14, 4 1982, issn: 15577341. doi: 10.1145/356893.
356896.

.. [6] F. Torres, J. Pomares, P. Gil, S. T. Puente y R. Aracil, Robots y Sistemas
Sensoriales. Madrid: Pearson Education, 2002, págs. 69-94.

.. [7] Z. Zhang, «A flexible new technique for camera calibration», IEEE Transac-
tions on Pattern Analysis and Machine Intelligence, vol. 22, 11 2000, issn:
01628828. doi: 10.1109/34.888718.

.. [8] R. I. Hartley, «Theory and practice of projective rectification», International
Journal of Computer Vision, vol. 35, 2 1999, issn: 09205691. doi: 10.1023/
A:1008115206617.

.. [9] D. Scharstein y R. Szeliski, «A taxonomy and evaluation of dense two-frame
stereo correspondence algorithms», International Journal of Computer Vi-
sion, vol. 47, 1-3 2002, issn: 09205691. doi: 10.1023/A:1014573219977.

.. [10] R. Szeliski, Computer Vision : Algorithms and Applications 2nd edition.
2020.

.. [11] S. Dai y W. Huang, A-TVSNet: Aggregated Two-View Stereo Network for
Multi-View Stereo Depth Estimation, 2020. arXiv: 2003.00711 [cs.CV].

.. [12] CS231n Convolutional Neural Networks for Visual Recognition, Accesado el
12, de febrero de 2021. dirección: https://cs231n.github.io/convolutional-
networks/.

.. [13] K. Kar, Mastering Computer Vision with TensorFlow 2.x. 2020.

.. [14] K. He, X. Zhang, S. Ren y J. Sun, Identity Mappings in Deep Residual
Networks, 2016. doi: 10 . 48550 / ARXIV . 1603 . 05027. dirección: https :
//arxiv.org/abs/1603.05027.

.. [15] R. Atienza, Advanced Deep Learning with Keras: Apply deep learning tech-
niques, autoencoders, GANs, variational autoencoders, deep reinforcement
learning, policy gradients, and more. 2018.

.. [16] J. Long, E. Shelhamer y T. Darrell, Fully Convolutional Networks for Se-
mantic Segmentation, 2014. doi: 10.48550/ARXIV.1411.4038. dirección:
https://arxiv.org/abs/1411.4038.

.. [17] ——, StereoPi V2 quick start guide, 2014. dirección: https://wiki.stereopi.
com/index.php?title=StereoPi_v2_Quick_Start_Guide.

.. [18] G. B. Adrian Kaehler, Learning OpenCV 3: computer vision in C++ with
the OpenCV library. California, USA: O’Reilly Media, Inc., 2016.

.. [19] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan y S. Belongie, Feature
Pyramid Networks for Object Detection, 2016. doi: 10.48550/ARXIV.1612.
03144. dirección: https://arxiv.org/abs/1612.03144.

.. [20] J. Redmon y A. Farhadi, «YOLOv3: An Incremental Improvement», arXiv,
2018.

.. [21] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn y A. Zisserman,
The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results,
http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html.

      