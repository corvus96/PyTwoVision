from logging import exception
import tensorflow as tf

from tensorflow.keras import Input 

from pytwovision.models.layers.conv2d_bn_leaky_relu_layer import Conv2dBNLeakyReluLayer
from pytwovision.models.layers.upsample_layer import UpsampleLayer


class BuildYoloV3Tiny(tf.keras.Model):
    def __init__(self, backbone, num_class):
        """Build YoloV3 tiny model given a backbone
        Arguments:
            name: a string with the name of the model
            backbone: an object with a backbone network
            num_class: an integer with the quantity of classes 
        Returns:
            A list where the first one is used to predict large-sized objects, 
            the second one is used to predict medium-sized objects and the  last one is the input shape returned.
        """
        super().__init__()
        if not isinstance(num_class, int):
            raise ValueError('num_class has to be an integer')
        self.base_outputs = backbone
        self.num_class = num_class
        
    def call(self, input_shape):
        input_shape = Input([input_shape.shape[0], input_shape.shape[1], input_shape.shape[2]])
        try:
            route_1, x = self.base_outputs.build_model(input_shape)
        except ValueError:
            raise Exception('Backbone output shape mismatch with yolov3 tiny input shape')

        x = Conv2dBNLeakyReluLayer((1, 1, 1024, 256))(x)

        conv_lobj_branch = Conv2dBNLeakyReluLayer((3, 3, 256, 512))(x)
        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255] (if num_class == 80 like coco dataset) 
        conv_lbbox = Conv2dBNLeakyReluLayer((1, 1, 512, 3*(self.num_class + 5)), activate=False, bn=False)(conv_lobj_branch)
                        
        x = Conv2dBNLeakyReluLayer((1, 1, 256, 128))(x)
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        x = UpsampleLayer()(x)
        x = tf.concat([x, route_1], axis=-1)

        conv_mobj_branch = Conv2dBNLeakyReluLayer((3, 3, 128, 256))(x)
        # conv_mbbox is used to predict medium size objects, shape = [None, 26, 26, 255] (if num_class == 80 like coco dataset) 
        conv_mbbox = Conv2dBNLeakyReluLayer((1, 1, 256, 3*(self.num_class + 5)), activate=False, bn=False)(conv_mobj_branch)

        return [conv_mbbox, conv_lbbox, input_shape]


        





