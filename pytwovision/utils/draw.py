import numpy as np
import cv2 as cv
import numpy as np
import colorsys
import random
import cv2 as cv

from pytwovision.utils.label_utils import read_class_names

def draw_lines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def draw_bbox(image, bboxes, class_file_name, show_label=True, show_confidence = True, text_colors=(255,255,0), rectangle_colors='', tracking=False, homogeneous_points=None):   
    """Draw bounding boxes on images

    Args:
        image: an array which correspond with an image
        bboxes: their bounding boxes.
        class_file_name: a path with a .txt file where the classes are saved.
        show_label: a boolean to show or hide object label.
        show_confidence: a boolean to show or hide confidence level.
        text_colors: a tuple that represents (R, G, B) colors.
        rectangle_colors: if this parameter is a string empty bounding box colors will be assing by default, however if rectangle_colors is a tuple like: (R, G, B) that will be bounding box colors.
        homogeneous_points: an array with dimensions n x 4 where each row is like (X, Y, Z, W). However if is None it won't be drawed.
        
    Returns:
        An image with bounding boxes and homogeneous coordinates.
    """
    classes = read_class_names(class_file_name)
    num_classes = len(classes)
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

            label = "{}".format(classes[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv.FILLED)

            # put text above rectangle
            cv.putText(image, label, (x1, y1-4), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, text_colors, bbox_thick, lineType=cv.LINE_AA)

        if homogeneous_points != None:
            y_position = y1 + 14
            coordinates = ["X", "Y:", "Z", "W"]
            for num, string_coor in enumerate(coordinates):
                coor_label = "{}: {:.2f}".format(string_coor, homogeneous_points[i][num])
                cv.putText(image, coor_label, (x1 + 3, y_position), cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale, text_colors, bbox_thick, lineType=cv.LINE_AA)
                y_position += 14


    return image