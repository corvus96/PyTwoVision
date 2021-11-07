from abc import ABCMeta, abstractmethod
from models.ssd_model import BuildSSD
from models.blocks.resnet_block import ResnetBlock, V1

class ModelManagerInterface(metaclass=ABCMeta):
    """
    An interface, which each method correspond with a network architecture.
    """
    @staticmethod
    @abstractmethod
    def build_SSD():
        """Implements simultaneous SSD."""


class ModelManager(ModelManagerInterface):
    """A selector of models that depends of which method is used."""
    def __init__(self):
        self.model1 = SSDModel()
    def build_SSD(self, name, backbone: ResnetBlock, n_layers=4, n_classes=4, aspect_ratios=(1, 2, 0.5)):
        """ Create the model to do SSD.
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
        return self.model1.build(name, backbone, n_layers, n_classes, aspect_ratios)

class SSDModel:
    def build(self, name, backbone, n_layers=4, n_classes=4, aspect_ratios=(1, 2, 0.5)):
        return BuildSSD(name, backbone, n_layers, n_classes, aspect_ratios)