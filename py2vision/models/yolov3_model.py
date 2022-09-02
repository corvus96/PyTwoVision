import tensorflow as tf

from tensorflow.keras import Input 

from py2vision.models.layers.conv2d_bn_leaky_relu_layer import conv2d_bn_leaky_relu_layer
from py2vision.models.layers.upsample_layer import UpsampleLayer


class BuildYoloV3(tf.keras.Model):
    """Build YoloV3 model given a backbone.
        
    Args:
        backbone: an object with a backbone network.
        num_class: an integer with the quantity of classes. 
        
    Returns:
        A list where the first one is used to predict large-sized objects, 
        the second one is used to predict medium-sized objects, the third one is used to small objects
        and the last one is the input shape returned.
    """
    def __init__(self, backbone, num_class):
        super().__init__()
        if not isinstance(num_class, int):
            raise ValueError('num_class has to be an integer')
        self.base_outputs = backbone
        self.num_class = num_class
        self.first_stack_filters = [(1, 1, 1024,  512), (3, 3,  512, 1024), (1, 1, 1024,  512), 
                                    (3, 3,  512, 1024), (1, 1, 1024,  512)]
        self.second_stack_filters = [(1, 1, 768, 256), (3, 3, 256, 512), (1, 1, 512, 256), 
                                    (3, 3, 256, 512), (1, 1, 512, 256)]
        self.third_stack_filters = [(1, 1, 384, 128), (3, 3, 128, 256), (1, 1, 256, 128),
                                    (3, 3, 128, 256), (1, 1, 256, 128)]
        
    def call(self, input_shape):
        if len(input_shape) != 3: raise ValueError("input shape should have a len == 3")
        input_shape = Input([input_shape[0], input_shape[1], input_shape[2]])
        try:
            route_1, route_2, x = self.base_outputs.build_model(input_shape)
        except ValueError:
            raise Exception('Backbone output shape mismatch with yolov3 input shape')

        for n, filters in enumerate(self.first_stack_filters):
            x = conv2d_bn_leaky_relu_layer(x, filters)
        
        conv_lobj_branch = conv2d_bn_leaky_relu_layer(x, (3, 3, 512, 1024))
        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255] (if num_class == 80 like coco dataset) 
        conv_lbbox = conv2d_bn_leaky_relu_layer(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)), activate=False, bn=False)
        
        x = conv2d_bn_leaky_relu_layer(x, (1, 1,  512,  256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        x = UpsampleLayer()(x)
        x = tf.concat([x, route_2], axis=-1)

        for n, filters in enumerate(self.second_stack_filters):
            x = conv2d_bn_leaky_relu_layer(x, filters)

        conv_mobj_branch = conv2d_bn_leaky_relu_layer(x, (3, 3, 256, 512))
        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255] (if num_class == 80 like coco dataset) 
        conv_mbbox = conv2d_bn_leaky_relu_layer(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)), activate=False, bn=False)
        
        x = conv2d_bn_leaky_relu_layer(x, (1, 1, 256, 128))
        x = UpsampleLayer()(x)
        x = tf.concat([x, route_1], axis=-1)

        for n, filters in enumerate(self.third_stack_filters):
            x = conv2d_bn_leaky_relu_layer(x, filters)
        
        conv_sobj_branch = conv2d_bn_leaky_relu_layer(x, (3, 3, 128, 256))
        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255] (if num_class == 80 like coco dataset) 
        conv_sbbox = conv2d_bn_leaky_relu_layer(conv_sobj_branch, (1, 1, 256, 3*(self.num_class +5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox, input_shape]


        





