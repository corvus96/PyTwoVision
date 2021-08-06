from __future__ import annotations
from abc import ABC, abstractmethod


class Recognizer:
    """
    The Abstraction defines the interface for the "control" part of the two
    class hierarchies. It maintains a reference to an object of the
    Implementation hierarchy and delegates all of the real work to this object.
    """

    def __init__(self, neural_network: NeuralNetwork):
        self.implementation = neural_network




class NeuralNetwork(ABC):
    """
    The Implementation defines the interface for all implementation classes. It
    doesn't have to match the Abstraction's interface. In fact, the two
    interfaces can be entirely different. Typically the Implementation interface
    provides only primitive operations, while the Abstraction defines higher-
    level operations based on those primitives.
    """

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def inference(self):
        pass
    
    @abstractmethod
    def restore_weights(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def print_summary(self):
        pass
    

    



