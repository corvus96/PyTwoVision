import os
import numpy as np
import tensorflow as tf

from recognition.selector import NeuralNetwork
from pytwovision.models.models_manager import ModelManager
from pytwovision.models.blocks.backbone_block import BackboneBlock
from pytwovision.models.blocks.backbone_block import darknet53, darknet19_tiny
from pytwovision.compute.yolov3_calculus import YoloV3Calculus

class ObjectDetectorYoloV3(NeuralNetwork):
    """Made of an ssd network model and a dataset generator.
    SSD defines functions to train and validate 
    an ssd network model.
    Arguments:
        backbone (class): Contains resnet network which can be version 1 or 2
        data_path (str): beginning by project root this is the path 
                         where is save your dataset; example: "dataset/drinks".
        train_labels_csv (str): contains the path of train labels csv.
        input_shape: A tuple with dims shape (height, weight, channels). 
        layers (int): Number of feature extraction layers of SSD head after backbone.
        threshold (float): Labels IoU threshold.
        normalize (bool): Use normalized predictions.   
        batch_size (int): Batch size during training.
        loss (str): Use focal and smooth L1 loss functions "focal-smooth-l1" 
        or smoth L1 "smooth-l1" even L1 "l1".
        class_threshold (float): Class probability threshold (>= is an object).
        iou_threshold (float): NMS IoU threshold.
        soft_nms (bool): Use soft NMS or not.
        save_dir_path: Directory for saving model and weights
    Attributes:
        ssd (model): SSD network model
        train_generator: Multi-threaded data generator for training
    """
    def __init__(self, model_name, num_class, data_path, train_labels_csv, input_shape=(416, 416, 3), yolo_version="yolov3", training=False, save_dir_path="weights", batch_size=4, threshold=0.6, normalize=False, class_threshold=0.5, iou_threshold=0.2, soft_nms=False) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_labels_csv = train_labels_csv
        self.batch_size = batch_size
        self.normalize = normalize
        self.threshold = threshold
        self.save_dir = save_dir_path
        self.class_threshold = class_threshold
        self.soft_nms = soft_nms
        self.iou_threshold = iou_threshold
        self.yolo = None
        self.train_generator = None
        self.num_class = num_class

        if yolo_version == "yolov3":
            backbone_net = BackboneBlock(darknet53())
            model_manager = ModelManager()
            self.conv_tensors = model_manager.build_yolov3(model_name, backbone_net, self.num_class)(input_shape)
        
        if yolo_version == "yolov3_tiny":
            backbone_net = BackboneBlock(darknet19_tiny())
            model_manager = ModelManager()
            self.conv_tensors = model_manager.build_yolov3_tiny(model_name, backbone_net, self.num_class)(input_shape)
        
        else: 
            versions = ["yolov3", "yolov3_tiny"]
            raise ValueError("yolo_version only can be: {}".format(", ".join(versions)))
        
        self.yolo = self.build_model(self.conv_tensors, training)

    
    def build_model(self, conv_tensors, training):
        """Build the complete yolo model and return model instance"""
        output_tensors = []
        input_layer = conv_tensors[-1]
        self.training = training
        compute = YoloV3Calculus()
        for i, conv_tensor in enumerate(conv_tensors[:-1]):
            pred_tensor = compute.decode(conv_tensor, self.num_class, i)
            if self.training: output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        return tf.keras.Model(input_layer, output_tensors)

    def build_dictionary(self):
        """Read input image filenames and obj detection labels
        from a csv file and store in a dictionary.
        """

        # build dictionary: 
        # key=image filaname, value=box coords + class label
        # self.classes is a list of class labels
        self.dictionary, self.classes = build_label_dictionary(self.train_labels_csv)
        self.n_classes = len(self.classes)
        self.keys = np.array(list(self.dictionary.keys()))
    
    def build_generator(self):
        """Build a multi-thread train data generator."""

        self.train_generator = SSDDataGenerator(input_shape=self.input_shape, 
                              batch_size=self.batch_size,
                              data_path=self.data_path,  
                              layers=self.layers,
                              threshold=self.threshold,
                              normalize=self.normalize,
                              dictionary=self.dictionary,
                              n_classes=self.n_classes,
                              feature_shapes=self.feature_shapes,
                              n_anchors=self.n_anchors,
                              shuffle=True)

    def train(self):
        """Train an ssd network.
        Arguments:
            epochs (int): Number of epochs to train.
            loss (str): Use focal and smooth L1 loss functions "focal-smooth-l1" 
            or smoth L1 "smooth-l1" even L1 "l1".
        """
        training = True
        if self.training == False:
            self.yolo = self.build_model(self.conv_tensors, training)



    def restore_weights(self, restore_weights):
        """Load previously trained model weights
        Arguments: 
            restore_weights (str): beginning by project root this is the path 
                                   where is save your weights; example: "weights/weights_01.h5"
        """
        if restore_weights:
            save_dir = os.path.join(os.getcwd(), self.save_dir)
            filename = os.path.join(save_dir, restore_weights)
            log = "Loading weights: %s" % filename
            print(log)
            self.ssd.load_weights(filename)


    def inference(self, image):
        """ Apply inference with trained model
        Arguments:
            image (tensor): contains an image converted in tensor.
        """
        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        image = np.squeeze(image, axis=0)
        classes = np.squeeze(classes)
        offsets = np.squeeze(offsets)
        return image, classes, offsets


    def evaluate(self, classes_names, image_file=None, image=None):
        """Evaluate image based on image (np tensor) or filename
        Arguments:
            classes_names (list): contains classes names asociated to classes index.
            image_file (str): contains the path of an image_file to evaluate the inference.
            image (tensor): contains an image converted in tensor to evalute  the inference.
        """
        show = False
        if image is None:
            image = skimage.img_as_float(imread(image_file))
            show = True

        image, classes, offsets = self.inference(image)
        class_names, rects, _, _ = show_boxes(image,
                                              classes,
                                              offsets,
                                              self.feature_shapes,
                                              classes_names,
                                              class_threshold=self.class_threshold,
                                              soft_nms=self.soft_nms,
                                              normalize=self.normalize,
                                              show=show,
                                              iou_threshold=self.iou_threshold)
        return class_names, rects


    def evaluate_test(self, data_path, test_labels_csv, classes_names):
        """ Apply evaluation in trained model.
        Arguments:
                data_path (str): folder that contains test files.
                test_labels_csv (str):  contains the path of test labels csv.
                classes_names (list): contains classes names asociated to classes index.
        """
        # test labels csv path
        path = test_labels_csv
        # test dictionary
        dictionary, _ = build_label_dictionary(path)
        keys = np.array(list(dictionary.keys()))
        # sum of precision
        s_precision = 0
        # sum of recall
        s_recall = 0
        # sum of IoUs
        s_iou = 0
        # evaluate per image
        for key in keys:
            # grounnd truth labels
            labels = np.array(dictionary[key])
            # 4 boxes coords are 1st four items of labels
            gt_boxes = labels[:, 0:-1]
            # last one is class
            gt_class_ids = labels[:, -1]
            # load image id by key
            image_file = os.path.join(data_path, key)
            image = skimage.img_as_float(imread(image_file))
            image, classes, offsets = self.inference(image)
            # perform nms
            _, _, class_ids, boxes = show_boxes(image,
                                              classes,
                                              offsets,
                                              self.feature_shapes,
                                              classes_names,
                                              class_threshold=self.class_threshold,
                                              soft_nms=self.soft_nms,
                                              normalize=self.normalize,
                                              show=False,
                                              iou_threshold=self.iou_threshold)

            boxes = np.reshape(np.array(boxes), (-1,4))
            # compute IoUs
            iou = SSDCalculus().iou(gt_boxes, boxes)
            # skip empty IoUs
            if iou.size ==0:
                continue
            # the class of predicted box w/ max iou
            maxiou_class = np.argmax(iou, axis=1)

            # true positive
            tp = 0
            # false positiove
            fp = 0
            # sum of objects iou per image
            s_image_iou = []
            for n in range(iou.shape[0]):
                # ground truth bbox has a label
                if iou[n, maxiou_class[n]] > 0:
                    s_image_iou.append(iou[n, maxiou_class[n]])
                    # true positive has the same class and gt
                    if gt_class_ids[n] == class_ids[maxiou_class[n]]:
                        tp += 1
                    else:
                        fp += 1

            # objects that we missed (false negative)
            fn = abs(len(gt_class_ids) - tp - fp)
            s_iou += (np.sum(s_image_iou) / iou.shape[0])
            s_precision += (tp/(tp + fp))
            s_recall += (tp/(tp + fn))


        n_test = len(keys)
        print("mIoU: %f" % (s_iou/n_test))
        print("Precision: %f" % (s_precision/n_test))
        print("Recall : %f" % (s_recall/n_test))


    def print_summary(self):
        """Print network summary for debugging purposes."""
        from tensorflow.keras.utils import plot_model
        self.backbone.summary()
        self.ssd.summary()
        plot_model(self.backbone,
                    to_file="backbone.png",
                    show_shapes=True)
