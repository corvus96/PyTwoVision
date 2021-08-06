"""Visualize bounding boxes
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from compute.ssd_calculus import SSDCalculus
from utils.label_utils import index2class, get_box_color

def show_boxes(image,
               classes,
               offsets,
               feature_shapes,
               class_threshold=0.5,
               normalize=False,
               show=True,
               soft_nms=False,
               iou_threshold=0.2):
    """Show detected objects on an image. Show bounding boxes
    and class names.
    Arguments:
        image (tensor): Image to show detected objects (0.0 to 1.0)
        classes (tensor): Predicted classes
        offsets (tensor): Predicted offsets
        feature_shapes (tensor): SSD head feature maps
        show (bool): Whether to show bounding boxes or not
    Returns:
        class_names (list): List of object class names
        rects (list): Bounding box rectangles of detected objects
        class_ids (list): Class ids of detected objects
        boxes (list): Anchor boxes of detected objects
    """
    # generate all anchor boxes per feature map
    anchors = []
    n_layers = len(feature_shapes)
    for index, feature_shape in enumerate(feature_shapes):
        anchor = SSDCalculus().anchor_boxes(feature_shape,
                              image.shape,
                              index=index)
        anchor = np.reshape(anchor, [-1, 4])
        if index == 0:
            anchors = anchor
        else:
            anchors = np.concatenate((anchors, anchor), axis=0)

    # get all non-zero (non-background) objects
    # objects = np.argmax(classes, axis=1)
    # print(np.unique(objects, return_counts=True))
    # nonbg = np.nonzero(objects)[0]
    if normalize:
        print("Normalize")
        anchors_centroid = SSDCalculus().minmax2centroid(anchors)
        offsets[:, 0:2] *= 0.1
        offsets[:, 0:2] *= anchors_centroid[:, 2:4]
        offsets[:, 0:2] += anchors_centroid[:, 0:2]
        offsets[:, 2:4] *= 0.2
        offsets[:, 2:4] = np.exp(offsets[:, 2:4])
        offsets[:, 2:4] *= anchors_centroid[:, 2:4]
        offsets = SSDCalculus().centroid2minmax(offsets)
        # convert fr cx,cy,w,h to real offsets
        offsets[:, 0:4] = offsets[:, 0:4] - anchors

    objects, indexes, scores = SSDCalculus().nms(classes,
                                   offsets,
                                   anchors,
                                   class_threshold=class_threshold,
                                   soft_nms=soft_nms,
                                   iou_threshold=iou_threshold)

    class_names = []
    rects = []
    class_ids = []
    boxes = []
    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(image)
    yoff = 1
    for idx in indexes:
        #batch, row, col, box
        anchor = anchors[idx] 
        offset = offsets[idx]
        
        anchor += offset[0:4]
        # default anchor box format is 
        # xmin, xmax, ymin, ymax
        boxes.append(anchor)
        w = anchor[1] - anchor[0]
        h = anchor[3] - anchor[2]
        x = anchor[0]
        y = anchor[2]
        category = int(objects[idx])
        class_ids.append(category)
        class_name = index2class(category)
        class_name = "%s: %0.2f" % (class_name, scores[idx])
        class_names.append(class_name)
        rect = (x, y, w, h)
        print(class_name, rect)
        rects.append(rect)
        if show:
            color = get_box_color(category)
            rect = Rectangle((x, y),
                             w,
                             h,
                             linewidth=2,
                             edgecolor=color,
                             facecolor='none')
            ax.add_patch(rect)
            bbox = dict(color='white',
                        alpha=1.0)
            ax.text(anchor[0] + 2,
                    anchor[2] - 16 + np.random.randint(0,yoff),
                    class_name,
                    color=color,
                    #fontweight='bold',
                    bbox=bbox,
                    fontsize=10,
                    verticalalignment='top')
            yoff += 50
            #t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

    if show:
        plt.savefig("detection.png", dpi=600)
        plt.show()

    return class_names, rects, class_ids, boxes


def show_anchors(image,
                 feature_shape,
                 anchors,
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
            class_name = index2class(category)
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