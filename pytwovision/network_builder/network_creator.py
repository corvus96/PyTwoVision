# Siamese
# desicion
# 2 Channel
# SPP
# Autoencoder
# Encoder
# Decoder
# ResNet
# CNN
# dnn o fcn

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Model

class NetworkCreator(metaclass=ABCMeta):
    """
    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    @staticmethod
    @abstractmethod
    def create_architecture():
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass


"""
Concrete Creators override the factory method in order to change the resulting
product's type.
"""


class BaseOnSiameseCreator:
    """
    Note that the signature of the method still uses the abstract product type,
    even though the concrete product is actually returned from the method. This
    way the Creator can stay independent of concrete product classes.
    """

    @staticmethod
    def create_architecture(net_name):
        "A static method to get a concrete product"
        if net_name == 'MatchNet':
            return MatchNet()
        return None

"""
Concrete Products provide various implementations of the Product interface.
"""
class MatchNet(NetworkCreator, Model):
    def __init__(self):
        super(MatchNet, self).__init__()
        filters = (24, 64, 96)
        kernel_size = (7, 3, 5)
        self.conv0 = Conv2D(filters[0], kernel_size[0], activation='relu')
        self.pool = MaxPooling2D(kernel_size[1], 2)
        self.conv1 = Conv2D(filters[1], kernel_size[2], activation='relu')
        self.conv2 = Conv2D(filters[2], kernel_size[1], activation='relu')
        self.conv3 = Conv2D(filters[1], kernel_size[1], activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    
    def create_architecture(self):
        return self

if __name__ == "__main__":
    print("App: Launched with the ConcreteCreator1.")
    model = BaseOnSiameseCreator.create_architecture('myModel')
    model.summary()
    print("\n")