from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
import tensorflow as tf
import time
import requests
import os 

from multiprocessing import Process, Queue
from py2vision.image_process.frame_decorator import Frame
from py2vision.image_process.resize_with_bbox import ResizeWithBBox
from py2vision.compute.yolov3_calculus import YoloV3Calculus
from py2vision.utils.draw import draw_bbox
from py2vision.input_output.camera import Camera

class DetectionMode(ABC):
    """
    The Abstract Class which defines detect method that contains the skeleton of detection algorithm with different methods such as detection on images, detection on video, detection using multiprocessing and real time detection.
    """
    def predict(self, model, image_data):
        """Apply a prediction step.

        Args:
            model: expects a tensorflow model trained.
            image_data: expects an array or tensor with image data.

        Returns:
            predicted bounding boxes.
        """
        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]

        return tf.concat(pred_bbox, axis=0)
    
    def postprocess_boxes(self, original_image, pred_bbox, input_size=416, score_threshold=0.3, iou_threshold=0.45, nms_method="nms"):
        """Apply a postprocess and non maximum supression step and resize bboxes.

        Args:
            original_image: expects a tensorflow model trained.
            pred_bbox: a tensor of predicted bounding boxes.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            nms_method: a string that can be  'nms' or 'soft-nms'.

        Returns:
            Improved bounding boxes
        """
        compute = YoloV3Calculus()
        bboxes = compute.postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)

        return compute.nms(bboxes, iou_threshold, method=nms_method)

    def pre_process(self, image, input_size):
        """Apply resize images step.

        Args:
            image: expects a an array.
            input_size: integer to resize an input image from their original dimensions to an square image.

        Returns:
            Resized images and bounding boxes.
        """
        frame = Frame(np.copy(image))
        image_data = ResizeWithBBox(frame).apply([input_size, input_size])

        return image_data[np.newaxis, ...].astype(np.float32)
    
    def draw(self, original_image, bboxes, class_file_name, rectangle_colors, homogeneous_points=None, text_colors=(255,255,0)):
        """Draw bounding boxes on images.

        Args:
            original_image: an array which correspond with an image
            bboxes: their bounding boxes.
            class_file_name: a path with a .txt file where the classes are saved.
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            homogeneous_points: an array with dimensions n x 4 where each row is like (X, Y, Z, W). However if is None it won't be drawed.

        Returns:
            An image with bounding boxes and homogeneous coordinates.
        """
        return draw_bbox(original_image, bboxes, class_file_name=class_file_name, 
                rectangle_colors=rectangle_colors, homogeneous_points=homogeneous_points, text_colors=text_colors)
    
    @abstractmethod
    def detect(self):
        """Apply detection pipeline"""
        pass
        
    @abstractmethod
    def prepare_input(self):
        """It's the first step to convert inputs like video paths or images in compatible data."""
        pass
    
    @abstractmethod
    def show(self):
        """Show actual frame in a window"""
        pass

    def camera_input(self, camera: Camera):
        """If camera source is webcam or other it is gonna create attribute 'vid', otherwise return a frame from streaming.
        
        Args:
            camera: is an instance of Camera Object
        """
        if camera.type_source == "other" or camera.type_source == "webcam":
            vid = cv.VideoCapture(camera.source)
            if not vid.isOpened():
                print("Cannot open camera")
                exit()
            return vid

        if camera.type_source == "stream":
            img_resp = requests.get(camera.source)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv.imdecode(img_arr, -1)
            return frame

class DetectRealTime(DetectionMode):
    def detect(self, model, camera: Camera, class_file_name, output_path="", input_size=416, 
                score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms", show=True):
        """Apply detection pipeline in realtime, if you want to close the window just press 'q'.
        
        Args:
            model: expects a tensorflow model trained.
            camera: An instance of camera object.
            class_file_name: it's the path of classes .txt file.
            output_path: if is an empty string, it won't be saved, but it is a path it save like a video.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm.
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
            show: a boolean to show frame pcessed.
        """
        times = []
        vid = self.camera_input(camera)
        
        if isinstance(vid, cv.VideoCapture):
            out, vid, fps = self.prepare_input(vid, output_path)
        
        while True:
            if isinstance(vid, cv.VideoCapture):
                ret, img = vid.read()
                if not ret: break
            else: 
                img = self.camera_input(camera)
            try:
                original_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                original_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2RGB)
            except:
                break
            image_data = self.pre_process(original_frame, input_size)
            t1 = time.time()
            pred_bbox = self.predict(model, image_data)
            t2 = time.time()
            bboxes = self.postprocess_boxes(original_frame, pred_bbox, input_size,
                                             score_threshold, iou_threshold, nms_method)
            times.append(t2-t1)
            times = times[-20:]
            
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            
            print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))
            frame = self.draw(original_frame, bboxes, class_file_name, rectangle_colors)
            cv.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                          cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if output_path != '': out.write(frame)
            if show: 
                self.show(frame, camera)
                if cv.waitKey(25) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    break
        if show: cv.destroyAllWindows()

    def prepare_input(self, vid, output_path):
        """Initialization for writing and reading videos
        
        Args: 
            vid: a cv.VideoCapture instance.
            output_path: if is an empty string, it won't be saved, but it is a path it save like an image.

        Returns: 
            a tuple where its first argument is an instance to Opencv video writes, the second element is an instance to read frames, and the last element is the frames per second
        """
        if os.path.splitext(output_path)[1] != ".mp4":
                raise ValueError("output_path has to be an .mp4 file")
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv.CAP_PROP_FPS))
        codec = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

        return out, vid, fps 
    
    def show(self, image, camera: Camera):
        """Show an image

        Args: 
            image: an image array.
            camera: An instance of camera object.
        """
        cv.imshow(camera.id, image)

class DetectRealTimeMP(DetectionMode):
    def detect(self, model, camera: Camera, class_file_name, output_path="", input_size=416, 
                score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms"):
        """Apply detection pipeline using multiprocessing, if you want to close the window just press 'q'.
        
        Args:
            model: expects a tensorflow model trained.
            camera: An instance of camera object.
            class_file_name: it's the path of classes .txt file
            output_path: if is an empty string, it won't be saved, but it is a path it save like a video.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
        """
        vid = self.camera_input(camera)

        if isinstance(vid, cv.VideoCapture):
            out, vid, fps = self.prepare_input(vid, output_path)
        original_frames = Queue()
        frames_data = Queue()
        predicted_data = Queue()
        processed_frames = Queue()
        processing_times = Queue()
        final_frames = Queue()
        p1, p2, p3 = self.multi_process_initialization(model, original_frames, frames_data, predicted_data, processed_frames, processing_times, final_frames, class_file_name, input_size, score_threshold, iou_threshold, rectangle_colors, nms_method)
        p1.start()
        p2.start()
        p3.start()
        while True:
            if isinstance(vid, cv.VideoCapture):
                ret, img = vid.read()
                if not ret: break
            else: 
                img = self.camera_input(camera)
            
            original_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
            original_frames.put(original_image)
            image_data = self.pre_process(original_image, input_size)
            frames_data.put(image_data)
        
        while True:
            if original_frames.qsize() == 0 and frames_data.qsize() == 0 and predicted_data.qsize() == 0  and processed_frames.qsize() == 0  and processing_times.qsize() == 0 and final_frames.qsize() == 0:
                p1.terminate()
                p2.terminate()
                p3.terminate()
                break
            elif final_frames.qsize()>0:
                image = final_frames.get()
                if output_path != '': out.write(image)

        cv.destroyAllWindows()

    def prepare_input(self, vid, output_path) -> None:
        """Initialization for writing and reading videos.

        Args: 
            vid: cv.VideoCapture instance
            output_path: if is an empty string, it won't be saved, but it is a path it save like an image.

        Returns: 
            a tuple where its first argument is an instance to Opencv video writes, the second element is an instance to read frames,
            and the last element is the frames per second
        """
        if os.path.splitext(output_path)[1] != ".mp4":
            raise ValueError("output_path has to be an .mp4 file")
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv.CAP_PROP_FPS))
        codec = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
        no_of_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

        return out, vid, fps 

    def predict_bbox_mp(self, model, frames_data, predicted_data, processing_times):
        """ predict bounding boxes using multiprocessing. It needs to be initialized.

        Args: 
            model: expects a tensorflow model trained.
            frames_data: a queue from multiprocessing package, that corresponds with frames. 
            predicted_data: a queue from multiprocessing package, that corresponds with predictions.
            processing_times: a queue from multiprocessing package, that corresponds with times.
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            try: tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
        times = []
        while True:
            if frames_data.qsize()>0:
                image_data = frames_data.get()
                t1 = time.time()
                processing_times.put(time.time())
                pred_bbox = self.predict(model, image_data)
                
                predicted_data.put(pred_bbox)
    
    def postprocess_mp(self, predicted_data, original_frames, processed_frames, processing_times, input_size, class_file_name, score_threshold, iou_threshold, rectangle_colors, nms_method):
        """ Improve bounding boxes using multiprocessing. It needs to be initialized.

        Args: 
            predicted_data: a queue from multiprocessing package, that corresponds with predictions.
            original_frames: a queue from multiprocessing package, that corresponds with frames. 
            processed_frames: a queue from multiprocessing package, that corresponds with processed frames. 
            processing times: a queue from multiprocessing package, that corresponds with times.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            class_file_name: it's the path of classes .txt file
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
        """
        times = []
        while True:
            if predicted_data.qsize()>0:
                pred_bbox = predicted_data.get()
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
                bboxes = self.postprocess_boxes(original_image, pred_bbox, input_size, score_threshold, iou_threshold, nms_method)
                image = self.draw(original_image, bboxes, class_file_name, rectangle_colors)
                times.append(time.time()-processing_times.get())
                times = times[-20:]
                
                ms = sum(times)/len(times)*1000
                fps = 1000 / ms
                image = cv.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                #print("Time: {:.2f}ms, Final FPS: {:.1f}".format(ms, fps))
                
                processed_frames.put(image)

    def show(self, processed_frames, final_frames):
        """ Show preocessed input.
        
        Args: 
            processed_frames: a queue from multiprocessing package, that corresponds with processed frames. 
            final_frames: a queue from multiprocessing package, that corresponds with final frames. 
        """
        while True:
            if processed_frames.qsize()>0:
                image = processed_frames.get()
                final_frames.put(image)
                cv.imshow('Detection output', image)
                if cv.waitKey(25) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    break

    def multi_process_initialization(self, model, original_frames, frames_data, predicted_data, processed_frames, processing_times, final_frames, class_file_name, input_size=416, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms"):
        """Apply an initialite multi processing prediction, postprocessing and show steps.
        
        Args:
            model: expects a tensorflow model trained.
            original_frames: a queue from multiprocessing package, that corresponds with frames. 
            frames_data: a queue from multiprocessing package, that corresponds with frames. 
            predicted_data: a queue from multiprocessing package, that corresponds with predictions.
            processed_frames: a queue from multiprocessing package, that corresponds with processed frames. 
            processing times: a queue from multiprocessing package, that corresponds with times.
            final_frames: a queue from multiprocessing package, that corresponds with final frames. 
            class_file_name: it's the path of classes .txt file
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.

        Returns: 
            an Process tuple (prediction, postprocessing, show)
        """
        p1 = Process(target=self.predict_bbox_mp, args=(model, frames_data, predicted_data, processing_times))
        p2 = Process(target=self.postprocess_mp, args=(predicted_data, original_frames, processed_frames, processing_times, input_size, class_file_name, score_threshold, iou_threshold, rectangle_colors, nms_method))
        p3 = Process(target=self.show, args=(processed_frames, final_frames))
        return p1, p2, p3


class DetectVideo(DetectionMode):
    def detect(self, model, input_path, class_file_name, output_path="", input_size=416, 
                score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms", show=True):
        """Apply detection pipeline in a saved video, if you want to close the window just press 'q'.
        
        Args:
            model: expects a tensorflow model trained.
            input_path: a video path.
            class_file_name: it's the path of classes .txt file
            output_path: if is an empty string, it won't be saved, but it is a path it save like a video.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
            show: a boolean to show frame pcessed
        """
        times, times_2 = [], []

        out, vid, fps = self.prepare_input(input_path, output_path)

        while True:
            _, img = vid.read()

            try:
                original_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
            except:
                break
            
            image_data = self.pre_process(original_image, input_size)
            t1 = time.time()
            pred_bbox = self.predict(model, image_data)
            t2 = time.time()
            bboxes = self.postprocess_boxes(original_image, pred_bbox, input_size, score_threshold, iou_threshold, nms_method)
            image = self.draw(original_image, bboxes, class_file_name, rectangle_colors)
            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)
            
            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
            
            image = cv.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
            
            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            if output_path != '': out.write(image)
            if show: 
                self.show(image)
                if cv.waitKey(25) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    break
        if show: cv.destroyAllWindows()

    def prepare_input(self, input_path, output_path):
        """Initialization for writing and reading videos.

        Args: 
            input_path: an video path.
            output_path: if is an empty string, it won't be saved, but it is a path it save like an image.
        
        Returns: 
            a tuple where its first argument is an instance to Opencv video writes, the second element is an instance to read frames,
            and the last element is the frames per second
        """
        if os.path.splitext(output_path)[1] != ".mp4":
            raise ValueError("output_path has to be an .mp4 file")
        vid = cv.VideoCapture(input_path)
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv.CAP_PROP_FPS))
        codec = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

        return out, vid, fps
    
    def show(self, image):
        """Show an image.
        Arguments: 
            image: an image array.
        """
        cv.imshow('Detection output', image)
        


class DetectImage(DetectionMode):
    def detect(self, model, input_path, class_file_name, output_path="", input_size=416,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', nms_method="nms", show=True):
        """Apply detection pipeline in an image, if you want to close the window just press 'q'.
        
        Arguments:
            model: expects a tensorflow model trained.
            input_path: an image path
            class_file_name: it's the path of classes .txt file
            output_path: if is an empty string, it won't be saved, but it is a path it save like an image.
            input_size: integer to resize bounding boxes from their resized dimensions to original dimensions (input_size).
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
            nms_method: a string that can be  'nms' or 'soft-nms'.
            show: a boolean to show frame pcessed.

        Returns:
            an image with prediction drawed
        """
        compatible_outputs = [".bmp", ".dib", ".jpg", ".jpeg", ".jpe", ".png", ".webp"]
        if not os.path.splitext(output_path)[1] in compatible_outputs: 
            raise ValueError("output_path only can be one of this: {}".format(compatible_outputs))
        original_image = self.prepare_input(input_path)
        image_data = self.pre_process(original_image, input_size)
        pred_bbox = self.predict(model, image_data)
        bboxes = self.postprocess_boxes(original_image, pred_bbox, input_size, score_threshold, iou_threshold, nms_method)
        image = self.draw(original_image, bboxes, class_file_name, rectangle_colors)
        if output_path != '': cv.imwrite(output_path, image)
        if show: self.show(image, output_path)

        return image

    def prepare_input(self, image_path):
        """Get a path an convert in an image array.

        Args: 
            image_path: a path.

        Returns: 
            An image array
        """
        original_image = cv.imread(image_path)
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

        return  cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    
    def show(self, image, output_path=''):
        """Get a an image, then save it and finally show it.

        Args: 
            image: an image array.
            output_path: if is an empty string, it won't be saved, but it is a path it save like an image.
        """
        # Show the image
        cv.imshow("predicted image", image)
        # Load and hold the image
        cv.waitKey(0)
        # To close the window after the required kill value was provided
        cv.destroyAllWindows()

        
