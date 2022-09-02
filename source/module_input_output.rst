=======
Inputs and Outputs
=======
Contains the classes that allow the input of data to the stereo and recognition system through external hardware such as cameras, video files or even live transmission via wifi, in addition to obtaining the intrinsic and extrinsic parameters of the physical medium that captured the images. On the other hand, its other function is to merge the capabilities of the recognition module and the stereo module with the VisionSystem class.

Inputs like
------------

.. automodule:: py2vision.input_output.camera
        :members:

How to calibrate a single camera?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    from py2vision.input_output.camera import Camera

    fisheye_camera = Camera("fisheye", "A_PATH_OR_A_NAME_FOR_CALIBRATION_IT_DOESN'T_MATTER")
    fisheye_camera.calibrate("A_CALIBRATION_IMAGE_FOLDER_PATH", show=False, export_file=False)

Outputs like
-------------
.. automodule:: py2vision.input_output.vision_system
        :members:
        
How to implement a position system?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    import os

    from py2vision.input_output.vision_system import VisionSystem
    from py2vision.input_output.camera import Camera
    from py2vision.stereo.standard_stereo import StandardStereo
    from py2vision.stereo.match_method import Matcher, StereoSGBM
    from py2vision.recognition.yolov3_detector import ObjectDetectorYoloV3
    from py2vision.recognition.selector import Recognizer

    anno_out_file = "annotations_formated"
    xml_path = "tests/test_dataset/annotations"
    classes_file = "tests/test_dataset/classes/coco.names"
    work_dir = "tests/test_dataset/to_generator_test"
    images_path = "tests/test_dataset/images"
    stereo_maps_path = "stereoMap"
    try:
        os.mkdir(work_dir)
    except:
        pass

    left_camera = Camera("left_camera", "tests/assets/photo/left/left_indoor_photo_5.png")
    right_camera = Camera("right_camera", "tests/assets/photo/right/right_indoor_photo_5.png")
    stereo_pair_fisheye = StandardStereo(left_camera, right_camera)
    stereo_pair_fisheye.calibrate("tests/assets/left_camera_calibration", "tests/assets/right_camera_calibration", show=False)
    stereo_pair_fisheye.rectify((640, 720), (640, 720), export_file=True, export_file_name=stereo_maps_path)
    # Add path format
    stereo_maps_path = stereo_maps_path + ".xml"
    sgbm = StereoSGBM(min_disp=-32, max_disp=32, window_size=3, p1=107, p2=710, pre_filter_cap=36, speckle_window_size=117, speckle_range=5, uniqueness_ratio=3, disp_12_max_diff=-38)
    matcher = Matcher(sgbm)
    lmbda = 13673
    sigma = 1.3175
    left_camera = Camera("left_camera", "tests/assets/photo/left/left_plant_1.png")
    right_camera = Camera("right_camera", "tests/assets/photo/right/right_plant_1.png")
    vis_sys = VisionSystem(left_camera, right_camera, stereo_maps_path, matcher, stereo_pair_fisheye.Q)
    yolov3 = ObjectDetectorYoloV3("test", 80, training=False)
    recognizer = Recognizer(yolov3)
    link_yolov3_weights = "https://pjreddie.com/media/files/yolov3.weights"
    wget.download(link_yolov3_weights)
    weights_file = os.path.basename(link_yolov3_weights)
    recognizer.restore_weights(weights_file)
    model = recognizer.get_model()
    # Here the magic happens
    vis_sys.image_pipeline(model, classes_file, os.path.join(work_dir, "test_position.jpg"),  lmbda=lmbda, sigma=sigma, downsample_for_match=None, show_window=True, score_threshold=0.5, iou_threshold=0.6, otsu_thresh_inverse=True, text_colors=(0, 0, 0))