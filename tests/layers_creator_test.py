from unittest import TestCase
from ..tests.pytwovision.network_builder.layers_creator import DownsampleLayer
 
 
class DownsampleLayerTest(TestCase):
    
    def setUp(self):
    ''' This method is execute before each test '''
        self.downsample = DownsampleLayer()
    
    def tearDown(self):
    ''' This method is execute after each test '''
        self.downsample.dispose()
 
    def test_stop(self):
 
        # self.car.speed = 5
 
        # self.car.stop()
 
        # # Verify the speed is 0 after stopping
 
        # self.assertEqual(0, self.car.speed)
 
         
 
        # # Verify it is Ok to stop again if the car is already stopped
 
        # self.car.stop()
 
        # self.assertEqual(0, self.car.speed)

if __name__ == '__main__':
    unittest.main()