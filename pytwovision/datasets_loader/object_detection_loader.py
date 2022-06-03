import pandas as pd
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf
import os
import io

from PIL import Image
from zoo_models.research.object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
from bs4 import BeautifulSoup
from utils.label_utils import class_text_to_int, label_map
# This line should be clear TEMPLATE METHOD IS THE PATTERN DESIGN
class ObjectDetectionLoader:
    def load(self, output_record_path, annotations_path, image_path, labels_path, csv_path=None):
        writer = tf.io.TFRecordWriter(output_record_path)
        path = os.path.join(image_path)
        annotations_csv = self.xml_to_csv(annotations_path)
        data = namedtuple('data', ['filename', 'object'])
        gb = annotations_csv.groupby('filename')
        grouped = [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
        for group in grouped:
            tf_example = self.create_tf_example(group, path, labels_path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord file: {}'.format(output_record_path))
        if csv_path is not None:
            annotations_csv.to_csv(csv_path, index=None)
            print('Successfully created the CSV file: {}'.format(csv_path))

    def create_tf_example(self, group, output_path, labels_path):
        with tf.io.gfile.GFile(os.path.join(output_path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        label_map = label_map_util.load_labelmap(labels_path)
        label_map_dict = label_map_util.get_label_map_dict(label_map)
        for index, row in group.object.iterrows():
            xmins.append(int(float(row['xmin'])) / width)
            xmaxs.append(int(float(row['xmax'])) / width)
            ymins.append(int(float(row['ymin'])) / height)
            ymaxs.append(int(float(row['ymax'])) / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class'], label_map_dict))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example
    
    def xml_to_csv(self, path):
        """Iterates through all .xml files in a given directory and combines
        them in a single Pandas dataframe.
        Args:
        path (str): The path containing the .xml files
        Returns
        -------
        Pandas DataFrame
            The produced dataframe
        """

        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            with open(xml_file, "r") as fp:
                contents = fp.read()
                soup = BeautifulSoup(contents,'xml')
                filename = soup.find('filename').contents[0]
                objects = soup.find_all('object')
                height = soup.find('height')
                width = soup.find('width')
                for obj in objects:
                    cls_name = obj.find('name').contents[0]
                    xmin = obj.find('xmin').contents[0]
                    xmax = obj.find('xmax').contents[0]
                    ymin = obj.find('ymin').contents[0]
                    ymax = obj.find('ymax').contents[0]
                    xml_list.append([filename, width, height, cls_name, xmin, ymin, xmax, ymax])
        column_name = ['filename', 'width', 'height',
                    'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df