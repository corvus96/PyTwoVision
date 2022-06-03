import unittest
import numpy as np
import os 
from pytwovision.utils.label_utils import label_map, index2class, class2index

class TestLabelUtils(unittest.TestCase):
    def setUp(self):
        self.labels = [{'name':'Mask', 'id':1}, {'name':'NoMask', 'id':2}]

    def test_label_map_none_path_is_the_same_folder(self):
        actual_folder = os.getcwd()
        test_name = 'test'
        test_path = actual_folder + "/{}.pbtxt".format(test_name)
        dst_path = label_map(self.labels, name=test_name)
        self.assertEqual(test_path, dst_path)

    def test_index2_class_and_class2_index_outputs(self):
        label_list = []
        for label in self.labels:
            label_list.append(label['name'])
        test_index = 0
        class_name = index2class(test_index, label_list)
        returned_index = class2index(class_name, label_list)
        self.assertEqual(test_index, returned_index)

        

if __name__ == '__main__':
    unittest.main()
