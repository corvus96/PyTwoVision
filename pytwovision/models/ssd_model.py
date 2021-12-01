import tensorflow as tf
import numpy as np

from models.blocks.resnet_block import ResnetBlock

from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class BuildSSD(tf.keras.Model):
    def __init__(self, name, backbone, n_layers=4, n_classes=4, aspect_ratios=(1, 2, 0.5)):
        """Build SSD model given a backbone
        Arguments:
            input_shape (list): input image shape
            backbone (model): Keras backbone model
            n_layers (int): Number of layers of ssd head
            n_classes (int): Number of obj classes
            aspect_ratios (list): annchor box aspect ratios
        Returns:
            n_anchors (int): Number of anchor boxes per feature pt
            feature_shape (tensor): SSD head feature maps
            model (Keras model): SSD model
        """
        super().__init__(name=name)
        # number of anchor boxes per feature map pt
        self.n_anchors = len(aspect_ratios) + 1

        # no. of base_outputs depends on n_layers
        self.base_outputs = backbone
        
        self.outputs = []
        self.feature_shapes = []
        self.out_cls = []
        self.out_off = []
        self.n_layers = n_layers
        self.n_classes = n_classes
        
    def call(self, input_shape):
        inputs = Input(shape=input_shape)
        # Backbone network
        x = self.base_outputs(inputs)

        for i in range(self.n_layers):
            # each conv layer from backbone is used
            # as feature maps for class and offset predictions
            # also known as multi-scale predictions
            conv = x if self.n_layers==1 else x[i]
            name = "cls" + str(i+1)
            conv_layer_A = Conv2D(filters=self.n_anchors * self.n_classes, kernel_size=3, name=name)
            classes = conv_layer_A(conv)

            # offsets: (batch, height, width, n_anchors * 4)
            name = "off" + str(i+1)
            conv_layer_B = Conv2D(filters=self.n_anchors*4, kernel_size=3, name=name)
            offsets  = conv_layer_B(conv)
            
            shape = np.array(K.int_shape(offsets))[1:]
            self.feature_shapes.append(shape)

            # reshape the class predictions, yielding 3D tensors of 
            # shape (batch, height * width * n_anchors, n_classes)
            # last axis to perform softmax on them
            name = "cls_res" + str(i+1)
            classes = Reshape((-1, self.n_classes), 
                            name=name)(classes)

            # reshape the offset predictions, yielding 3D tensors of
            # shape (batch, height * width * n_anchors, 4)
            # last axis to compute the (smooth) L1 or L2 loss
            name = "off_res" + str(i+1)
            offsets = Reshape((-1, 4),
                            name=name)(offsets)
            # concat for alignment with ground truth size
            # made of ground truth offsets and mask of same dim
            # needed during loss computation
            offsets = [offsets, offsets]
            name = "off_cat" + str(i+1)
            offsets = Concatenate(axis=-1,
                                name=name)(offsets)

            # collect offset prediction per scale
            self.out_off.append(offsets)

            name = "cls_out" + str(i+1)

            #activation = 'sigmoid' if n_classes==1 else 'softmax'
            #print("Activation:", activation)

            classes = Activation('softmax',
                                name=name)(classes)

            # collect class prediction per scale
            self.out_cls.append(classes)

        if self.n_layers > 1:
            # concat all class and offset from each scale
            name = "offsets"
            offsets = Concatenate(axis=1,
                                name=name)(self.out_off)
            name = "classes"
            classes = Concatenate(axis=1,
                                name=name)(self.out_cls)
        else:
            offsets = self.out_off[0]
            classes = self.out_cls[0]

        self.outputs = [classes, offsets]
        model = Model(inputs=inputs,
                    outputs=self.outputs,
                    name='ssd_head')

        return self.n_anchors, self.feature_shapes, model