from __future__ import annotations
from abc import ABC, abstractmethod

import glob
import os
import xml.etree.ElementTree as ET


class Parser(ABC):
    """
    The Parser interface declares an `parse` method that should take the
    base AnnotationsFormat interface as an argument.
    """

    @abstractmethod
    def parse(self, anno: AnnotationsFormat):
        pass


class XmlParser(Parser):
    """
    Each Concrete Parser must implement the `parse` method in such a way
    that it calls the annotationsFormat's method corresponding to the Parser's class.
    """

    def parse(self, anno: AnnotationsFormat, xml_path, annotations_output_name, classes_output_name, work_dir=None):
        """
        This method convert annotations  from COCO or PASCAL VOC dataset in xml format
         to be compatible with an especific network model. Exporting a 
         text file for annotations and a text file for classes names 
         Arguments:
            xml_path: a string with the full path of xml annotations.
            annotations_output_name: a string with the name of annotations file that will be generated.
            classes_output_name: a string with the name of classes file that will be generated.
            work_dir: a path where the annotations and classes files will be saved, if is None these will be 
            saved in current directory.
        """

        anno.visit_xml_parser(self, xml_path, annotations_output_name, classes_output_name, work_dir)

class AnnotationsFormat(ABC):
    """
    The AnnotationsFormat Interface declares a set of visiting methods that correspond to
    Parser classes. The signature of a visiting method allows the visitor to
    identify the exact class of the Parser that it's dealing with.
    """

    @abstractmethod
    def visit_xml_parser(self, element: XmlParser, xml_path, annotations_output_name, classes_output_name, work_dir=None):
        pass

class YoloV3AnnotationsFormat(AnnotationsFormat):
    def visit_xml_parser(self, element ,xml_path, annotations_output_name, classes_output_name, work_dir=None):
        xmls = glob.glob(xml_path+'/*.xml')
        if len(xmls) == 0:
            raise FileNotFoundError("There isn't annotations in {}".format(xml_path))
        if work_dir is None:
            anno_file = os.getcwd()
            anno_file = os.path.join(anno_file, annotations_output_name)
            classes_file = os.getcwd()
            classes_file = os.path.join(classes_file, classes_output_name)
        else:
            anno_file = os.path.join(work_dir, annotations_output_name)
            classes_file = os.path.join(work_dir, classes_output_name)
        classes_names = []
        with open('{}.txt'.format(anno_file), 'w') as file:
            for xml_file in xmls:
                tree = ET.parse(open(xml_file))
                root = tree.getroot()
                image_name = root.find('filename').text
                img_path = xml_path + '/' + image_name
                for i, obj in enumerate(root.iter('object')):
                    cls = obj.find('name').text
                    if cls not in classes_names:
                        classes_names.append(cls)
                    cls_id = classes_names.index(cls)
                    xmlbox = obj.find('bndbox')
                    object_params = (str(int(float(xmlbox.find('xmin').text)))+','
                            +str(int(float(xmlbox.find('ymin').text)))+','
                            +str(int(float(xmlbox.find('xmax').text)))+','
                            +str(int(float(xmlbox.find('ymax').text)))+','
                            +str(cls_id))
                    img_path += ' '+object_params
                print(img_path)
                file.write(img_path+'\n')
        
        with open('{}.txt'.format(classes_file), 'w') as file:
            for name in classes_names:
                file.write(str(name)+'\n')