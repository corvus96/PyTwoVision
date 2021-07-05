from abc import ABCMeta, abstractmethod
from models.blocks.resnet_blocks import ResnetBackbone
from models.blocks.simultaneous_heads import Heads
from models.blocks.pretrain_simultaneous_heads import PretrainHead

class ModelManagerInterface(metaclass=ABCMeta):
    """
    An interface, which each method correspond with a network architecture.
    """
    @staticmethod
    @abstractmethod
    def simultaneous_OD_SS():
        """Implements simultaneous object detection and semantic segmentation model."""


class ModelManager(ModelManagerInterface):
    """A selector of models that depends of which method is used."""
    def __init__(self):
        self.model1 = ObjectDetectSemanticSegmentationNet()
    def simultaneous_OD_SS(self, name, pretrain = False, backboneNet = True, config = None):
        """ Create the model to do simultaneous  object detection and semantic segmentation.
        
        Attributes:
            name: Name of the model
            pretrain: is a Boolean, which if it is true the last layers in Resnetbackbone will have a dilation_rate of 1.Otherwise dilation rate will be greather than 1.
            backboneNet: is a Boolean, which if is true create only backbone network that works as a feature extraction net. Otherwise it will create two heads network for object detection and semantic segmentation.
            config: is an object,

        """
        return self.model1.build(name, pretrain, backboneNet, config)

class ObjectDetectSemanticSegmentationNet:
    def build(self, name, pretrain, backboneNet, config) -> None:
        if backboneNet:
            return ResnetBackbone(name, pretrain)
        else:
            return Heads(name, config)


if __name__ == "__main__":
    # The client code.
    
    manager = ModelManager()
    #manager.simultaneous_OD_SS('backbone')
    #manager.simultaneous_OD_SS('heads', backboneNet= False)