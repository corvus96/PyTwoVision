=======
Datasets loader
=======
It was developed because the way in which the data sets introduced in a network are presented can vary depending on the type of recognition (detection, classification or segmentation) and even the training mechanism of each network architecture varies, sometimes requiring special preprocessing. For these reasons in this module each file must contain a class that adapts the data set to be introduced in a neural network. Currently it hosts the **YoloV3DatasetGenerator** class, which takes the annotations of a file containing the locations of the images of the set, its bounding boxes and the class to which the object belongs and adapts them to the network for training and evaluation. 
    
    .. automodule:: pytwovision.datasets_loader.yolov3_dataset_generator
        :members: