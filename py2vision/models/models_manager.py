from abc import ABCMeta, abstractmethod
from py2vision.models.yolov3_model import BuildYoloV3
from py2vision.models.yolov3_tiny_model import BuildYoloV3Tiny

class ModelManagerInterface(metaclass=ABCMeta):
    """
    An interface, which each method correspond with a network architecture.
    """
    @staticmethod
    @abstractmethod
    def build_yolov3():
        """Implements yolov3."""

    @staticmethod
    @abstractmethod
    def build_yolov3_tiny():
        """Implements yolov3 tiny"""


class ModelManager(ModelManagerInterface):
    """A selector of models that depends of which method is used.

    Attributes:
        model1: an instance of Yolov3Model
        model2: an instance of Yolov3TinyModel
    """
    def __init__(self):
        self.model1 = Yolov3Model()
        self.model2 = Yolov3TinyModel()
    
    def build_yolov3(self, backbone, num_class):
        """ Create the model to do YoloV3 based in darknet53.

        Args:
            backbone: an object with a backbone network.
            num_class: an integer with the quantity of classes.

        Returns:
            A list where the first one is used to predict large-sized objects, the second one is used to predict medium-sized objects, the third one is used to small objects and the last one is the input shape returned.
        """
        return self.model1.build(backbone, num_class)
    
    def build_yolov3_tiny(self, backbone, num_class):
        """ Create the model to do YoloV3 based in darknet19 tiny.

        Args:
            backbone: an object with a backbone network.
            num_class: an integer with the quantity of classes.

        Returns:
            A list where the first one is used to predict large-sized objects, the second one is used to predict medium-sized objects and the  last one is the input shape returned.
        """
        return self.model2.build(backbone, num_class)

class Yolov3Model:
    def build(self, backbone, num_class):
        return BuildYoloV3(backbone, num_class)

class Yolov3TinyModel:
    def build(self, backbone, num_class):
        return BuildYoloV3Tiny(backbone, num_class)