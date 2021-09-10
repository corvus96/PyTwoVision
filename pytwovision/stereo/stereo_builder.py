from __future__ import annotations
from abc import ABC, abstractmethod

class StereoSystemBuilder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def pre_process(self) -> None:
        pass

    @abstractmethod
    def match(self) -> None:
        pass

    @abstractmethod
    def post_process(self) -> None:
        pass


class StereoController:
    """
    The Director is only responsible for executing the building steps in a
    particular sequence. It is helpful when producing products according to a
    specific order or configuration. Strictly speaking, the Director class is
    optional, since the client can control builders directly.
    """

    def __init__(self) -> None:
        self._stereo_builder = None

    @property
    def stereo_builder(self) -> StereoSystemBuilder:
        return self._stereo_builder

    @stereo_builder.setter
    def stereo_builder(self, builder: StereoSystemBuilder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        """
        self._stereo_builder = builder

    """
    The Director can construct several product variations using the same
    building steps.
    """

    def make_minimal_viable_product(self) -> None:
        self.stereo_builder.produce_part_a()

    def make_full_featured_product(self) -> None:
        self.stereo_builder.produce_part_a()
        self.stereo_builder.produce_part_b()
        self.stereo_builder.produce_part_c()