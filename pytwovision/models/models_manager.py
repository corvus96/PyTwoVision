from abc import ABCMeta, abstractmethod
from models.blocks.resnet_blocks import ResnetBackbone
from models.blocks.simultaneous_heads import Heads
from models.blocks.pretrain_simultaneous_heads import PretrainHead

class ModelManagerInterface(metaclass=ABCMeta):
    """
    The ModelManager interface declares all the methods to implement.
    """
    @staticmethod
    @abstractmethod
    def simultaneous_OD_SS():
        """A method to implement simultaneous object detection and semantic segmentation
        made by Niels Ole Salscheider
        """


class ModelManager(ModelManagerInterface):
    """A selector of models that depends of which method is used."""
    def __init__(self):
        self.model1 = ObjectDetectSemanticSegmentationNet()
    def simultaneous_OD_SS(self, name, pretrain = False, backboneNet = True, config = None):
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
    manager.simultaneous_OD_SS('backbone')
    #manager.simultaneous_OD_SS('heads', backboneNet= False)