"""Label utility functions
Main use: labeling, dictionary of colors,
label retrieval, loading label csv file,
drawing label on an image
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import os

from matplotlib.patches import Rectangle
from random import randint

def get_box_color(index=None):
    """Retrieve plt-compatible color string based on object index"""
    colors = ['w', 'r', 'b', 'g', 'c', 'm', 'y', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]


def get_box_rgbcolor(index=None):
    """Retrieve rgb color based on object index"""
    colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 128, 0)]
    if index is None:
        return colors[randint(0, len(colors) - 1)]
    return colors[index % len(colors)]

def label_map(labels, dst_path=None, name='label_map'):
    """
    An easy way to convert classes names and ids to a 
    .pbtxt file compatible with tensorflow models API.
    Args:
        labels (list): a list, which each element is a dictionary with two keys
        'name' and 'id'.
        dst_path (str): a path where the file will be saved.
        name (str): the name of the file, with which it will save the .pbtxt
    Returns:
        a string where the file was saved.
    Raises:
        Exception: When someone element in internal dictionaries have another keys different 
        of name and id.
        TypeError: when labels aren't list type
        ValueError: when labels are empty
    """
    if type(labels) != list: raise TypeError('labels should be a list of dictionaries!')
    if not labels: raise ValueError('labels are empty')
    if dst_path == None:
        dst_path = os.getcwd()
    file_path = dst_path +  '/{}.pbtxt'.format(name)
    allowed_keys = ['name', 'id']
    with open(file_path, 'w') as f:
        for label in labels:
            if allowed_keys == list(label.keys()):
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')
            else: 
                os.remove(file_path)
                raise Exception("Your keys should be only 'name' and 'id'")
        print('file saved in: {}'.format(file_path))
    
    return file_path
            
def index2class(index, classes: list):
    """Convert index (int) to class name (string)"""
    return classes[index]


def class2index(class_name, classes: list):
    """Convert class name (string) to index (int)"""
    return classes.index(class_name)

def class_text_to_int(row_label, label_map_dict: dict):
    return label_map_dict[row_label]

def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
        
def load_csv(path):
    """Load a csv file into an np array"""
    data = []
    with open(path) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            data.append(row)

    return np.array(data)

def get_label_dictionary(labels, keys):
    """Associate key (filename) to value (box coords, class)"""
    dictionary = {}
    for key in keys:
        dictionary[key] = [] # empty boxes

    for label in labels:
        if len(label) != 6:
            print("Incomplete label:", label[0])
            continue

        value = label[1:]

        if value[0]==value[1]:
            continue
        if value[2]==value[3]:
            continue

        if label[-1]==0:
            print("No object labelled as bg:", label[0])
            continue

        # box coords are float32
        value = value.astype(np.float32)
        # filename is key
        key = label[0]
        # boxes = bounding box coords and class label
        boxes = dictionary[key]
        boxes.append(value)
        dictionary[key] = boxes

    # remove dataset entries w/o labels
    for key in keys:
        if len(dictionary[key]) == 0:
            del dictionary[key]

    return dictionary


def build_label_dictionary(path):
    """Build a dict with key=filename, value=[box coords, class]"""
    labels = load_csv(path)
    # skip the 1st line header
    labels = labels[1:]
    # keys are filenames
    keys = np.unique(labels[:,0])
    dictionary = get_label_dictionary(labels, keys)
    classes = np.unique(labels[:,-1]).astype(int).tolist()
    # insert background label 0
    # classes.insert(0, 0)
    print("Num of unique classes: ", classes)
    return dictionary, classes


def show_labels(image, labels, ax=None):
    """Draw bounding box on an object given box coords (labels[1:5])"""
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
    for label in labels:
        # default label format is xmin, xmax, ymin, ymax
        w = label[1] - label[0]
        h = label[3] - label[2]
        x = label[0]
        y = label[2]
        category = int(label[4])
        color = get_box_color(category)
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y),
                         w,
                         h,
                         linewidth=2,
                         edgecolor=color,
                         facecolor='none')
        ax.add_patch(rect)
    plt.show()