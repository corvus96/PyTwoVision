from __future__ import annotations
from abc import ABC, abstractmethod

class StereoSystemBuilder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @property
    @abstractmethod
    def get_product(self):
        pass

    @abstractmethod
    def pre_process(self):
        pass

    @abstractmethod
    def match(self):
        pass

    @abstractmethod
    def post_process(self):
        pass
    def find_epilines(self):
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

    def compute_disparity(self, frameL, frameR):
        rectify_left, rectify_right = self.stereo_builder.pre_process(frameL, frameR)
        left_disp, right_disp, left_matcher = self.stereo_builder.match(rectify_left, rectify_right)
        return self.stereo_builder.post_process(frameL, left_disp, right_disp, left_matcher)
    def compute_disparity_without_refinement(self, frameL, frameR):
        rectify_left, rectify_right = self.stereo_builder.pre_process(frameL, frameR)
        return self.stereo_builder.match(rectify_left, rectify_right)
    def make_epilines(self, frameL, frameR):
        return self.stereo_builder.find_epilines(frameL, frameR)

    def make_full_featured_product(self) -> None:
        self.stereo_builder.produce_part_a()
        self.stereo_builder.produce_part_b()
        self.stereo_builder.produce_part_c()