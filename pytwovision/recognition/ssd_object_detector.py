import os
import numpy as np
import skimage

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from skimage.io import imread

from recognition.selector import Recognizer
from recognition.selector import NeuralNetwork
from models.ssd_model import BuildSSD
from models.hiperparameters_config.learning_rate_ssd_scheduler import lr_scheduler
from datasets_loader.ssd_dataset_generator import SSDDataGenerator
from utils.label_utils import build_label_dictionary
from utils.boxes import show_boxes
from models.loss.detection_loss_utils import focal_loss_categorical, smooth_l1_loss, l1_loss
from compute.ssd_calculus import SSDCalculus
from models.blocks.resnet_block import ResnetBlock

class ObjectDetectorSSD(NeuralNetwork):
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
    def __init__(self, backbone: ResnetBlock, data_path, train_labels_csv, input_shape=(480, 640, 3), save_dir_path="weights", layers=4, batch_size=4, threshold=0.6, normalize=False, class_threshold=0.5, iou_threshold=0.2, soft_nms=False) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_labels_csv = train_labels_csv
        self.batch_size = batch_size
        self.layers =layers
        self.normalize = normalize
        self.threshold = threshold
        self.save_dir = save_dir_path
        self.class_threshold = class_threshold
        self.soft_nms = soft_nms
        self.iou_threshold = iou_threshold
        self.ssd = None
        self.train_generator = None
        # input shape is (480, 640, 3) by default
        self.input_shape = (input_shape[0], 
                            input_shape[1],
                            input_shape[2])
        
        # store in a dictionary the list of image files and labels
        self.build_dictionary()

        # build the backbone network (for example ResNet50)
        # the number of feature layers is equal to n_layers
        # feature layers are inputs to SSD network heads
        # for class and offsets predictions
        self.backbone = backbone.build_model(self.input_shape, n_layers=self.layers)

        # using the backbone, build ssd network
        # outputs of ssd are class and offsets predictions
        model = BuildSSD('SSD Model', self.backbone, n_layers=self.layers, n_classes=self.n_classes)
        anchors, features, ssd = model(self.input_shape)
        # n_anchors = num of anchors per feature point (for example 4)
        self.n_anchors = anchors
        # feature_shapes is a list of feature map shapes
        # per output layer - used for computing anchor boxes sizes
        self.feature_shapes = features
        # ssd network model
        self.ssd = ssd
    
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

        self.train_generator = \
                SSDDataGenerator(input_shape=self.input_shape, 
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


    def train(self, epochs=200,  loss_function="l1"):
        """Train an ssd network.
        Arguments:
            epochs (int): Number of epochs to train.
            loss (str): Use focal and smooth L1 loss functions "focal-smooth-l1" 
            or smoth L1 "smooth-l1" even L1 "l1".
        """
        # build the train data generator
        if self.train_generator is None:
            self.build_generator()

        optimizer = Adam(lr=1e-3)
        # choice of loss functions 
        if loss_function == "focal-smooth-l1":
            print("Focal loss and smooth L1")
            loss = [focal_loss_categorical, smooth_l1_loss]
        elif loss_function == "smooth-l1":
            print("Smooth L1")
            loss = ['categorical_crossentropy', smooth_l1_loss]
        else:
            print("Cross-entropy and L1")
            loss = ['categorical_crossentropy', l1_loss]

        self.ssd.compile(optimizer=optimizer, loss=loss)

        # model weights are saved for future validation
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), self.save_dir)
        model_name = self.backbone.name
        model_name += '-' + str(self.layers) + "layer"
        if self.normalize:
            model_name += "-norm"
        if loss_function == "focal-smooth-l1":
            model_name += "-improved_loss"
        elif loss_function == "smooth-l1":
            model_name += "-smooth_l1"

        if self.threshold < 1.0:
            model_name += "-extra_anchors" 

        model_name += "-" 
        # Get dataset filename
        root, _ = os.path.splitext(self.data_path)
        _, dataset = os.path.split(root)
        model_name += dataset
        model_name += '-{epoch:03d}.h5'

        log = "# of classes %d" % self.n_classes
        print(log)
        log = "Batch size: %d" % self.batch_size
        print(log)
        log = "Weights filename: %s" % model_name
        print(log)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # prepare callbacks for saving model weights
        # and learning rate scheduler
        # learning rate decreases by 50% every 20 epochs
        # after 60th epoch
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     verbose=1,
                                     save_weights_only=True)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [checkpoint, scheduler]
        # train the ssd network
        self.ssd.fit(self.train_generator,
                     use_multiprocessing=False,
                     callbacks=callbacks,
                     epochs=epochs)


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
