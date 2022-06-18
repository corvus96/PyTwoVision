"""Visualize bounding boxes
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import colorsys
import random
import cv2 as cv

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from compute.ssd_calculus import SSDCalculus
from utils.label_utils import index2class, get_box_color
from pytwovision.utils.label_utils import read_class_names

def show_boxes(image, bboxes, classes_file_path, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    classes_names = read_class_names(classes_file_path)
    num_classes = len(classes_names)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(classes_names[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv.FILLED)

            # put text above rectangle
            cv.putText(image, label, (x1, y1-4), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv.LINE_AA)

    return image


def show_anchors(image,
                 feature_shape,
                 anchors,
                 classes_names=['car', 'person', 'dog', 'chair', 'boat', 'sofa', 'tvmonitor', 'bird', 'aeroplane', 'cow', 'cat', 'pottedplant', 'bottle', 'horse', 'bicycle', 'motorbike', 'diningtable', 'bus', 'train', 'sheep'],
                 maxiou_indexes=None,
                 maxiou_per_gt=None,
                 labels=None,
                 show_grids=False):
    """Utility for showing anchor boxes for debugging purposes"""
    image_height, image_width, _ = image.shape
    _, feature_height, feature_width, _ = feature_shape

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if show_grids:
        grid_height = image_height // feature_height
        for i in range(feature_height):
            y = i * grid_height
            line = Line2D([0, image_width], [y, y])
            ax.add_line(line)

        grid_width = image_width // feature_width
        for i in range(feature_width):
            x = i * grid_width
            line = Line2D([x, x], [0, image_height])
            ax.add_line(line)

    # maxiou_indexes is (4, n_gt)
    for index in range(maxiou_indexes.shape[1]):
        i = maxiou_indexes[1][index]
        j = maxiou_indexes[2][index]
        k = maxiou_indexes[3][index]
        # color = label_utils.get_box_color()
        box = anchors[0][i][j][k] #batch, row, col, box
        # default anchor box format is xmin, xmax, ymin, ymax
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = box[0]
        y = box[2]
        # Rectangle ((xmin, ymin), width, height) 
        rect = Rectangle((x, y),
                         w,
                         h,
                         linewidth=2,
                         edgecolor='y',
                         facecolor='none')
        ax.add_patch(rect)

        if maxiou_per_gt is not None and labels is not None:
            # maxiou_per_gt[index] is row w/ max iou
            iou = np.amax(maxiou_per_gt[index])
            #argmax_index = np.argmax(maxiou_per_gt[index])
            #print(maxiou_per_gt[index])
            # offset
            label = labels[index]
            category = int(label[4])
            class_name = index2class(category, classes_names)
            color = get_box_color(category)
            bbox = dict(facecolor=color, color=color, alpha=1.0)
            ax.text(label[0],
                    label[2],
                    class_name,
                    color='w',
                    fontweight='bold',
                    bbox=bbox,
                    fontsize=16,
                    verticalalignment='top')
            dxmin = label[0] - box[0]
            dxmax = label[1] - box[1]
            dymin = label[2] - box[2]
            dymax = label[3] - box[3]
            print(index, ":", "(", class_name, ")", iou, dxmin, dxmax, dymin, dymax, label[0], label[2])

    if labels is None:
        plt.show()

    return fig, ax