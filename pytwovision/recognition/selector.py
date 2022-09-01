from __future__ import annotations
from abc import ABC, abstractmethod

import re
import wget
import os



class Recognizer:
    """
    An Abstraction that expects an implementation of a detector like:
    'ObjectDetectorYoloV3'.
    """

    def __init__(self, neural_network: NeuralNetwork):
        self.implementation = neural_network

    def get_model(self):
        """Returns a tensorflow model with an implemented network"""
        return self.implementation.model
    
    def print_model(self):
        """Print network summary for debugging purposes."""
        self.implementation.print_summary()

    def inference(self, image_path, input_size=416, score_threshold=0.3, iou_threshold=0.45, nms_method="nms"):
        """ Apply inference with trained model.

        Args:
            image_path: a path to an image.
            input_size: integer to resize an input image from their original dimensions to an square image.
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm
            nms_method: a string that can be  'nms' or 'soft-nms'.

        Returns:
            An array with bounding boxes.
        """
        return self.implementation.inference(image_path, input_size, score_threshold, iou_threshold, nms_method)

    def restore_weights(self, weights_file, use_checkpoint=False):
        """Load previously trained model weights.

        Args: 
            weights_file: beginning by project root this is the path where is save your weights; example: "weights/weights_01.h5".
            use_checkpoint: if you wanna use a .ckpt file this variable should be True.
        """
        self.implementation.restore_weights(weights_file, use_checkpoint)

    def train_using_weights(self, train_annotations_path, test_annotations_path, class_file_name, weights_path,
                checkpoint_path="checkpoints", use_checkpoint=False, warmup_epochs=2, 
                epochs=100, log_dir="logs", save_only_best_model=True, save_all_checkpoints=False, batch_size=4, lr_init=1e-4, lr_end=1e-6,
                strides=[8, 16, 32],
                anchors=[[[10,  13], [16,   30], [33,   23]],
                        [[30,  61], [62,   45], [59,  119]],
                        [[116, 90], [156, 198], [373, 326]]],
                anchor_per_scale=3, max_bbox_per_scale=100):
        """Train with transfer learning.
        
        Args:
            train_annotations_path: a string corresponding to the folder where train annotations are located.
            test_annotations_path: a string corresponding to the folder where test annotations are located.
            class_file_name: a string corresponding to the classes file (a .txt file with a list of classes) is located.
            weights_path: a path which in case if it's an url with weights like: 'https://pjreddie.com/media/files/yolov3.weights' or 'https://pjreddie.com/media/files/yolov3-tiny.weights' this method first download the weights and then train the net but if the path is a local file the method is going to load the weights and next train the network.
            checkpoint_path: a string corresponding to the checkpoint file that is inside of a checkpoints folder.
            use_checkpoint: a boolean that controls if use chepoint before train.
            warmup_epochs: an hiperparameter that update learning rate like this paper https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg.
            epochs: Number of epochs to train.
            log_dir: a folder to save logs.
            save_only_best_model: if is true the model will be saved when best validation loss > total validation loss/total test elements, but if it isn't true model will be saved always.
            save_all_checkpoints: it is a boolean, if is true model will be saved in each epoch.
            batch_size: an integer with the size of batches in test and train datasets.
            lr_init: a float which is initial learning rate.
            lr_end: a float which is final learning rate.
            strides: a list with the strides in a yolo model.
            anchors: these are the yolo anchors sizes.
            anchor_per_scale: an integer with the number of anchor boxes per scale. 
            max_bbox_per_scale: nan integer with the number of bounding boxes per scale. 
        """
        regex = '((http|https)://)(www.)?' + '[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]' + '{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)'
        regular_exp = re.compile(regex)
        if re.search(regular_exp, weights_path):
            wget.download(weights_path)
            weights_path = os.path.basename(weights_path)
        # restore weights
        self.implementation.restore_weights(weights_path)
        # train with a pre-trained net
        self.implementation.train(train_annotations_path, test_annotations_path, class_file_name, checkpoint_path, use_checkpoint, warmup_epochs, epochs, log_dir, save_only_best_model, save_all_checkpoints, batch_size, lr_init, lr_end, strides=strides, anchors=anchors, anchor_per_scale=anchor_per_scale, max_bbox_per_scale=max_bbox_per_scale)

    def train(self, train_annotations_path, test_annotations_path, class_file_name, 
                checkpoint_path="checkpoints", use_checkpoint=False, warmup_epochs=2, 
                epochs=100, log_dir="logs", save_only_best_model=True, save_all_checkpoints=False, batch_size=4, lr_init=1e-4, lr_end=1e-6,
                strides=[8, 16, 32],
                anchors=[[[10,  13], [16,   30], [33,   23]],
                        [[30,  61], [62,   45], [59,  119]],
                        [[116, 90], [156, 198], [373, 326]]],
                anchor_per_scale=3, max_bbox_per_scale=100):
        """Train an ssd network.

        Args:
            train_annotations_path: a string corresponding to the folder where train annotations are located.
            test_annotations_path: a string corresponding to the folder where test annotations are located.
            class_file_name: a string corresponding to the classes file (a .txt file with a list of classes) is located.
            checkpoint_path: a string corresponding to the checkpoint file that is inside of a checkpoints folder.
            use_checkpoint: a boolean that controls if use chepoint before train. 
            warmup_epochs: an hiperparameter that update learning rate like this paper https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg.
            epochs: Number of epochs to train.
            log_dir: a folder to save logs.
            save_only_best_model: if is true the model will be saved when best validation loss > total validation loss/total test elements, but if it isn't true model will be saved always.
            save_all_checkpoints: it is a boolean, if is true model will be saved in each epoch.
            batch_size: an integer with the size of batches in test and train datasets.
            lr_init: a float which is initial learning rate.
            lr_end: a float which is final learning rate.
            strides: a list with the strides in a yolo model.
            anchors: these are the yolo anchors sizes.
            anchor_per_scale: an integer with the number of anchor boxes per scale. 
            max_bbox_per_scale: nan integer with the number of bounding boxes per scale. 
        """
        self.implementation.train(train_annotations_path, test_annotations_path, class_file_name, checkpoint_path, use_checkpoint, warmup_epochs, epochs, log_dir, save_only_best_model, save_all_checkpoints, batch_size, lr_init, lr_end, strides=strides, anchors=anchors, anchor_per_scale=anchor_per_scale, max_bbox_per_scale=max_bbox_per_scale)
    
    def evaluate(self, model, dataset, classes_file, score_threshold=0.05, iou_threshold=0.50, test_input_size=416):
        """Apply evaluation using mAP.

        Args:
            model: a tesorflow detection model.
            dataset: an YoloV3DatasetGenerator instance with test dataset.
            classes_file: a string corresponding to the classes file (a .txt file with a list of classes) is located.
            score_threshold: if the score of a bounding boxes is less than score_threshold, it will be discard.
            iou_threshold: a parameter between (0, 1) which is used for nms algorithm.
            test_input_size: integer to resize an input image from their original dimensions to an square image.

        Returns:
            mAP score
        """
        return self.implementation.evaluate(model, dataset, classes_file, score_threshold, iou_threshold, test_input_size)



class NeuralNetwork(ABC):
    """
    The Implementation defines the interface for all implementation classes. It doesn't have to match the Abstraction's interface. In fact, the two interfaces can be entirely different. Typically the Implementation interface provides only primitive operations, while the Abstraction defines higher-level operations based on those primitives.
    """

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def inference(self):
        pass
    
    @abstractmethod
    def restore_weights(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def print_summary(self):
        pass
    

    



