import unittest
import numpy as np

from pytwovision.models.blocks.backbone_block import BackboneBlock
from pytwovision.models.blocks.backbone_block import darknet53, darknet19_tiny


class TestBackboneBlock(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(2000)
        self.input_data = np.random.rand(3,4,4,1)

    def test_darknet53_output(self):
        model = BackboneBlock(darknet53())
        route1, route2, output = model.build_model(self.input_data)
        self.assertEqual(output.shape, (3, 0, 0, 1024))
        self.assertEqual(route1.shape, (3, 0, 0, 256))
        self.assertEqual(route2.shape, (3, 0, 0, 512))

    def test_darknet19_tiny_output(self):
        model = BackboneBlock(darknet19_tiny())
        route1, output = model.build_model(self.input_data)
        self.assertEqual(output.shape, (3, 1, 1, 1024))
        self.assertEqual(route1.shape, (3, 1, 1, 256))


if __name__ == '__main__':
    unittest.main()
