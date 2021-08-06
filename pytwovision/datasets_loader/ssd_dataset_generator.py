from tensorflow.keras import layers
from tensorflow.python.keras.utils.data_utils import Sequence

import numpy as np
import os
import skimage

from compute.ssd_calculus import SSDCalculus

from skimage.io import imread



class SSDDataGenerator(Sequence):
    """Multi-threaded data generator.
    Each thread reads a batch of images and their object labels
    Arguments:
        data_path (str): beginning by project root this is the path 
                        where is save your dataset; example: "dataset/drinks"
        dictionary: Dictionary of image filenames and object labels 
                    where key=image filaname, value=box coords + class label
        layers (int): Number of feature extraction layers of SSD head after backbone
        threshold (float): Labels IoU threshold
        normalize (bool): Use normalized predictions
        batch_size (int): Batch size during training
        input_shape: A tuple with dims shape (height, weight, channels)
        n_classes (int): Number of object classes
        feature_shapes (tensor): Shapes of ssd head feature maps
        n_anchors (int): Number of anchor boxes per feature map pt
        shuffle (Bool): If dataset should be shuffled 
    """
    def __init__(self,
                 data_path,
                 dictionary,
                 n_classes, 
                 layers=4,
                 threshold=0.6,
                 normalize=False,
                 batch_size=4,
                 input_shape=(480, 640, 3),
                 feature_shapes=[],
                 n_anchors=4,
                 shuffle=True):
        self.batch_size = batch_size
        self.data_path = data_path
        self.layers = layers
        self.normalize = normalize
        self.threshold = threshold
        self.dictionary = dictionary
        self.n_classes = n_classes
        self.keys = np.array(list(self.dictionary.keys()))
        self.input_shape = input_shape
        self.feature_shapes = feature_shapes
        self.n_anchors = n_anchors
        self.shuffle = shuffle
        self.on_epoch_end()
        self.get_n_boxes()


    def __len__(self):
        """Number of batches per epoch"""
        blen = np.floor(len(self.dictionary) / self.batch_size)
        return int(blen)


    def __getitem__(self, index):
        """Get a batch of data"""
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        keys = self.keys[start_index : end_index]
        x, y = self.__data_generation(keys)
        return x, y


    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.keys)


    def get_n_boxes(self):
        """Total number of bounding boxes"""
        self.n_boxes = 0
        for shape in self.feature_shapes:
            self.n_boxes += np.prod(shape) // self.n_anchors
        return self.n_boxes

    def __data_generation(self, keys):
        """Generate train data: images and 
        object detection ground truth labels 
        Arguments:
            keys (array): Randomly sampled keys
                (key is image filename)
        Returns:
            x (tensor): Batch images
            y (tensor): Batch classes, offsets, and masks
        """
        # train input data
        x = np.zeros((self.batch_size, *self.input_shape))
        dim = (self.batch_size, self.n_boxes, self.n_classes)
        # class ground truth
        gt_class = np.zeros(dim)
        dim = (self.batch_size, self.n_boxes, 4)
        # offsets ground truth
        gt_offset = np.zeros(dim)
        # masks of valid bounding boxes
        gt_mask = np.zeros(dim)

        for i, key in enumerate(keys):
            # images are assumed to be stored in self.data_path
            # key is the image filename 
            image_path = os.path.join(self.data_path, key)
            image = skimage.img_as_float(imread(image_path))
            # assign image to a batch index
            x[i] = image
            # a label entry is made of 4-dim bounding box coords
            # and 1-dim class label
            labels = self.dictionary[key]
            labels = np.array(labels)
            # 4 bounding box coords are 1st four items of labels
            # last item is object class label
            boxes = labels[:,0:-1]
            for index, feature_shape in enumerate(self.feature_shapes):
                # generate anchor boxes
                anchors = SSDCalculus().anchor_boxes(feature_shape,
                                       image.shape,
                                       index=index,
                                       n_layers=self.layers)
                # each feature layer has a row of anchor boxes
                anchors = np.reshape(anchors, [-1, 4])
                # compute IoU of each anchor box 
                # with respect to each bounding boxes
                iou = SSDCalculus().iou(anchors, boxes)

                # generate ground truth class, offsets & mask
                gt = SSDCalculus().get_gt_data(iou,
                                 n_classes=self.n_classes,
                                 anchors=anchors,
                                 labels=labels,
                                 normalize=self.normalize,
                                 threshold=self.threshold)
                gt_cls, gt_off, gt_msk = gt
                if index == 0:
                    cls = np.array(gt_cls)
                    off = np.array(gt_off)
                    msk = np.array(gt_msk)
                else:
                    cls = np.append(cls, gt_cls, axis=0)
                    off = np.append(off, gt_off, axis=0)
                    msk = np.append(msk, gt_msk, axis=0)

            gt_class[i] = cls
            gt_offset[i] = off
            gt_mask[i] = msk


        y = [gt_class, np.concatenate((gt_offset, gt_mask), axis=-1)]

        return x, y