import tensorflow as tf

class BuildSSD(tf.keras.Model):
    def __init__(self, input_shape, backbone, n_layers=4, n_classes=4, aspect_ratios=(1, 2, 0.5)):
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

