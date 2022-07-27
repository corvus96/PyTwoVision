import cv2 as cv
import numpy as np
import time
import os

from pytwovision.input_output.camera import Camera
from pytwovision.recognition.detection_mode import DetectRealTime
from pytwovision.recognition.detection_mode import DetectImage
from pytwovision.stereo.match_method import Matcher
from pytwovision.stereo.stereo_builder import StereoController
from pytwovision.stereo.standard_stereo import StandardStereoBuilder
from pytwovision.compute.yolov3_calculus import YoloV3Calculus
from pytwovision.image_process.frame_decorator import Frame
from pytwovision.image_process.resize import Resize

class VisionSystem:
    """Provide an interface to apply recognition and stereo vision"""
    def __init__(self, cam_left: Camera, cam_right: Camera, stereo_maps_path, matcher: Matcher, q_matrix):
        """Initialization of all necessary parameteres to implement an stereo-recognition system 
        Arguments:
            cam_left: a camera instance which can be streaming, mp4 file path, image path or even realtime source.
            cam_right: a camera instance which can be streaming, mp4 file path, image path or even realtime source.
            stereo_maps_path: 
            q_matrix (arr): a 4x4 array with the following structure, 
                    [[1 0   0          -cx     ]
                     [0 1   0          -cy     ]
                     [0 0   0           f      ]
                     [0 0 -1/Tx (cx - cx')/Tx ]]
                    cx: is the principal point x in left image
                    cx': is the principal point x in right image
                    cy: is the principal point y in left image
                    f: is the focal lenth in left image
                    Tx: The x coordinate in Translation matrix
        """
        self.camL = cam_left
        self.camR = cam_right
        stereo_builder = StandardStereoBuilder(self.camL, self.camR, stereo_maps_path)
        self.stereo_controller = StereoController()
        self.stereo_controller.stereo_builder = stereo_builder
        self.matcher = matcher
        self.Q = q_matrix

    def realtime_or_video_pipeline(self, model, class_file_name, output_path="", input_size=416, 
                score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms", 
                post_process_match=True, lmbda=128.0, sigma=1.5, downsample_for_match=2, show_window=True):
        """ Implement a stereo recognition system for video or streaming 
        Arguments:
            model: expects a tensorflow model trained.
            class_file_name: it's the path of classes .txt file
            output_path: if is an empty string, it won't be saved, but it is a path it save like a video.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default,
            however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
            post_process_match: if is true apply post_process and return improved disparity map, 
            otherwise return left disparity map without post processing.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step.
            downsample_for_match: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor 
            just can be 2, 4, 8, 16, 32 or 64. If downsample factor is 1 or None or False won't apply downsampling.
            show_window: show a window with implementation.
        """
        times, times_2 = [], []
        detector = DetectRealTime()
        inputL = detector.camera_input(self.camL)
        inputR = detector.camera_input(self.camR)
        if isinstance(inputL, cv.VideoCapture):
            if downsample_for_match:
                temp_inputL = inputL.read()
                temp_inputL = cv.pyrDown(temp_inputL)
                out, _, _ = detector.prepare_input(temp_inputL, output_path)
            else:
                out, inputL, _ = detector.prepare_input(inputL, output_path)
        compute = YoloV3Calculus()

        while True:
            if hasattr(self, "vid"):
                retL, imgL = inputL.read()
                retR, imgR = inputR.read()
                if not retL: break
                if not retR: break
            else: 
                imgL = self.camera_input(self.camL)
                imgR = self.camera_input(self.camR)
            try:
                frameL = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
                frameL = cv.cvtColor(frameL, cv.COLOR_BGR2RGB)
            except:
                break
            # Preprocess
            left_for_matcher, right_for_matcher = self.stereo_controller.pre_process_step(imgL, imgR, downsample_for_match)
            left_for_detector = detector.pre_process(frameL, input_size)
            # match and predict
            t1 = time.time()
            left_disp, right_disp, left_matcher = self.stereo_controller.stereo_builder.match(left_for_matcher, 
                                                                                                right_for_matcher, self.matcher, False)
            pred_bbox = detector.predict(model, left_for_detector)
            t2 = time.time()
            # post_process
            if post_process_match:
                disparity = self.stereo_controller.stereo_builder.post_process(left_for_matcher, left_disp, right_disp, left_matcher, lmbda=lmbda, sigma=sigma, metrics=False)
            else: 
                disparity = left_disp
            # recover original size
            if downsample_for_match in [1, None, False]:
                    n_upsamples = 0
            else:
                n_upsamples = [2**p for p in range(1, 7)].index(downsample_for_match)
            if n_upsamples > 0:
                for i in range(n_upsamples + 1):
                    disparity = cv.pyrUp(disparity)
            # resize bboxes
            bboxes = detector.postprocess_boxes(disparity, pred_bbox, input_size,
                                             score_threshold, iou_threshold, nms_method)
            # get distance
            center_bboxes = compute.minmax2centroid(bboxes[..., :4])
            points_3D = self.stereo_controller.stereo_builder.estimate_3D_points(center_bboxes[..., :2], disparity, self.Q)
            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)
            times = times[-20:]
            times_2 = times_2[-20:]
            
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
            # draw on image
            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            frame = detector.draw(frameL, bboxes, class_file_name, rectangle_colors, homogeneous_points=points_3D)

            cv.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                          cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if output_path != '': out.write(frame)
            if show_window:
                detector.show(frame, self.camL)
            if cv.waitKey(25) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break
        cv.destroyAllWindows()

    def image_pipeline(self, model, class_file_name, output_path="", input_size=416, 
                score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms", 
                post_process_match=True, lmbda=128.0, sigma=1.5, downsample_for_match=2, show_window=True):
        """ Implement a stereo recognition system for images.
        Arguments:
            model: expects a tensorflow model trained.
            class_file_name: it's the path of classes .txt file
            output_path: if is an empty string, it won't be saved, but it is a path it save like a video.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default,
            however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
            post_process_match: if is true apply post_process and return improved disparity map, 
            otherwise return left disparity map without post processing.
            lmbda: is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000. Only valid in post processing step
            sigma: is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0. Only valid in post processing step.
            downsample_for_match: if it is true, it will apply blurry in both frames and downsamples it. The downsampling factor 
            just can be 2, 4, 8, 16, 32 or 64. If downsample factor is 1 or None or False won't apply downsampling.
            show_window: show a window with implementation.
        Returns:
            an image processed, bounding boxes and 3D points
        """
        detector = DetectImage()
        compute = YoloV3Calculus()
        compatible_outputs = [".bmp", ".dib", ".jpg", ".jpeg", ".jpe", ".png", ".webp"]
        if not os.path.splitext(output_path)[1] in compatible_outputs: 
            raise ValueError("output_path only can be one of this: {}".format(compatible_outputs))
        imgL = detector.prepare_input(self.camL.source)
        imgR = detector.prepare_input(self.camR.source)
        # pre-process
        left_for_matcher, right_for_matcher = self.stereo_controller.pre_process_step(imgL, imgR, downsample_for_match)
        left_for_detector = detector.pre_process(imgL, input_size)
        # match and predict
        left_disp, right_disp, left_matcher = self.stereo_controller.stereo_builder.match(left_for_matcher, 
                                                                                            right_for_matcher, self.matcher, False)
        pred_bbox = detector.predict(model, left_for_detector)
        # postprocess
        if post_process_match:
            disparity = self.stereo_controller.stereo_builder.post_process(left_for_matcher, left_disp, right_disp, left_matcher, lmbda=lmbda, sigma=sigma, metrics=False)
        else: 
            disparity = left_disp
        # recover original size
        if downsample_for_match in [1, None, False]:
                n_upsamples = 0
        else:
            n_upsamples = [2**p for p in range(1, 7)].index(downsample_for_match)
        if n_upsamples > 0:
            for i in range(n_upsamples + 1):
                disparity = cv.pyrUp(disparity)
        # resize bboxes
        bboxes = detector.postprocess_boxes(disparity, pred_bbox, input_size,
                                             score_threshold, iou_threshold, nms_method)
        # get distance
        center_bboxes = compute.minmax2centroid(bboxes[..., :4])
        points_3D = self.stereo_controller.stereo_builder.estimate_3D_points(center_bboxes[..., :2], disparity, self.Q)
        # draw on image
        frame = detector.draw(imgL, bboxes, class_file_name, rectangle_colors, homogeneous_points=points_3D)
        if output_path != '': cv.imwrite(output_path, frame)
        if show_window:
            # Show the image
            cv.imshow(self.camL.id, frame)
            # Load and hold the image
            cv.waitKey(0)
            # To close the window after the required kill value was provided
            cv.destroyAllWindows()

        return frame, bboxes, points_3D
