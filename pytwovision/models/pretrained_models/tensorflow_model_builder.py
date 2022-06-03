import wget
import tarfile
import os
import tensorflow as tf

from zoo_models.research.object_detection.utils import config_util
from zoo_models.research.object_detection.protos import pipeline_pb2
from google.protobuf import text_format

class TensorflowModel():
    def __init__(self, link) -> None:
        DOWNLOAD_LINK_ASSERT = 'http://download.tensorflow.org/models'
        if DOWNLOAD_LINK_ASSERT in link: is_valid_model = True 
        else: is_valid_model = False
        # Download a model from tensorflow zoo
        if is_valid_model:
            print('downloading: {}'.format(link))
            wget.download(link)
        else:
            raise ValueError('This is not a tensorflow model, please download a model from: {}'.format(DOWNLOAD_LINK_ASSERT))
        # Uncompress the model
        file_name = os.path.basename(link)
        print('\nextracting elements from: {}'.format(file_name))
        file = tarfile.open(file_name)
        file.extractall('.')
        file.close()
        os.remove(file_name)
        self.name = file_name[:file_name.index('.tar.gz')]
        print('{} was removed'.format(file_name))
    
    def config_parameters(self, model_path, parameters_file):
        model_config_path = os.path.join(model_path, self.name, parameters_file)
        config = config_util.get_configs_from_pipeline_file(model_config_path)
        return config


#if __name__ == '__main__':
    #TensorflowModel('http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz')