import numpy as np
import math
import tensorflow as tf

class YoloV3Calculus:
    """
    Utility methods for computing IOU, anchor boxes, masks,
    and bounding box offsets
    """

    def decode(self, conv_output, num_class, i=0, strides=[8, 16, 32], 
                anchors=[[[10,  13], [16,   30], [33,   23]],
                        [[30,  61], [62,   45], [59,  119]],
                        [[116, 90], [156, 198], [373, 326]]]):
        strides = np.array(strides)
        anchors = (np.array(anchors).T/strides).T
        # where i = 0, 1 or 2 to correspond to the three grid scales  
        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size,dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def centroid2minmax(self, boxes):
        """Centroid to minmax format 
        (cx, cy, w, h) to (xmin, xmax, ymin, ymax)
        Arguments:
            boxes (tensor): Batch of boxes in centroid format
        Returns:
            minmax (tensor): Batch of boxes in minmax format
        """
        minmax= np.copy(boxes).astype(np.float)
        minmax[..., 0] = boxes[..., 0] - (0.5 * boxes[..., 2])
        minmax[..., 1] = boxes[..., 0] + (0.5 * boxes[..., 2])
        minmax[..., 2] = boxes[..., 1] - (0.5 * boxes[..., 3])
        minmax[..., 3] = boxes[..., 1] + (0.5 * boxes[..., 3])
        return minmax

    def minmax2centroid(self, boxes):
        """Minmax to centroid format
        (xmin, xmax, ymin, ymax) to (cx, cy, w, h)
        Arguments:
            boxes (tensor): Batch of boxes in minmax format
        Returns:
            centroid (tensor): Batch of boxes in centroid format
        """
        centroid = np.copy(boxes).astype(np.float)
        centroid[..., 0] = 0.5 * (boxes[..., 1] - boxes[..., 0])
        centroid[..., 0] += boxes[..., 0] 
        centroid[..., 1] = 0.5 * (boxes[..., 3] - boxes[..., 2])
        centroid[..., 1] += boxes[..., 2] 
        centroid[..., 2] = boxes[..., 1] - boxes[..., 0]
        centroid[..., 3] = boxes[..., 3] - boxes[..., 2]
        return centroid

    def intersection(self, boxes1, boxes2):
        """Compute intersection of batch of boxes1 and boxes2
        
        Arguments:
            boxes1 (tensor): Boxes coordinates in pixels
            boxes2 (tensor): Boxes coordinates in pixels
        Returns:
            intersection_areas (tensor): intersection of areas of
                boxes1 and boxes2
        """
        m = boxes1.shape[0] # The number of boxes in `boxes1`
        n = boxes2.shape[0] # The number of boxes in `boxes2`

        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

        boxes1_min = np.expand_dims(boxes1[:, [xmin, ymin]], axis=1)
        boxes1_min = np.tile(boxes1_min, reps=(1, n, 1))
        boxes2_min = np.expand_dims(boxes2[:, [xmin, ymin]], axis=0)
        boxes2_min = np.tile(boxes2_min, reps=(m, 1, 1))
        min_xy = np.maximum(boxes1_min, boxes2_min)

        boxes1_max = np.expand_dims(boxes1[:, [xmax, ymax]], axis=1)
        boxes1_max = np.tile(boxes1_max, reps=(1, n, 1))
        boxes2_max = np.expand_dims(boxes2[:, [xmax, ymax]], axis=0)
        boxes2_max = np.tile(boxes2_max, reps=(m, 1, 1))
        max_xy = np.minimum(boxes1_max, boxes2_max)

        side_lengths = np.maximum(0, max_xy - min_xy)

        intersection_areas = side_lengths[:, :, 0] * side_lengths[:, :, 1]
        return intersection_areas

    def union(self, boxes1, boxes2, intersection_areas):
        """Compute union of batch of boxes1 and boxes2
        Arguments:
            boxes1 (tensor): Boxes coordinates in pixels
            boxes2 (tensor): Boxes coordinates in pixels
        Returns:
            union_areas (tensor): union of areas of
                boxes1 and boxes2
        """
        m = boxes1.shape[0] # number of boxes in boxes1
        n = boxes2.shape[0] # number of boxes in boxes2

        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

        width = (boxes1[:, xmax] - boxes1[:, xmin])
        height = (boxes1[:, ymax] - boxes1[:, ymin])
        areas = width * height
        boxes1_areas = np.tile(np.expand_dims(areas, axis=1), reps=(1,n))
        width = (boxes2[:,xmax] - boxes2[:,xmin])
        height = (boxes2[:,ymax] - boxes2[:,ymin])
        areas = width * height
        boxes2_areas = np.tile(np.expand_dims(areas, axis=0), reps=(m,1))

        union_areas = boxes1_areas + boxes2_areas - intersection_areas
        return union_areas

    def iou(self, boxes1, boxes2):
        """Compute IoU of batch boxes1 and boxes2
        Arguments:
            boxes1 (tensor): Boxes coordinates in pixels
            boxes2 (tensor): Boxes coordinates in pixels
        Returns:
            iou (tensor): intersectiin of union of areas of
                boxes1 and boxes2
        """
        intersection_areas = self.intersection(boxes1, boxes2)
        union_areas = self.union(boxes1, boxes2, intersection_areas)
        return intersection_areas / union_areas
    
    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area

    def get_gt_data(self, iou,
                n_classes=4,
                anchors=None,
                labels=None,
                normalize=False,
                threshold=0.6):
        """Retrieve ground truth class, bbox offset, and mask
        
        Arguments:
            iou (tensor): IoU of each bounding box wrt each anchor box
            n_classes (int): Number of object classes
            anchors (tensor): Anchor boxes per feature layer
            labels (list): Ground truth labels
            normalize (bool): If normalization should be applied
            threshold (float): If less than 1.0, anchor boxes>threshold
                are also part of positive anchor boxes
        Returns:
            gt_class, gt_offset, gt_mask (tensor): Ground truth classes,
                offsets, and masks
        """
        # each maxiou_per_get is index of anchor w/ max iou
        # for the given ground truth bounding box
        maxiou_per_gt = np.argmax(iou, axis=0)
        
        # get extra anchor boxes based on IoU
        if threshold < 1.0:
            iou_gt_thresh = np.argwhere(iou>threshold)
            if iou_gt_thresh.size > 0:
                extra_anchors = iou_gt_thresh[:,0]
                extra_classes = iou_gt_thresh[:,1]
                #extra_labels = labels[:,:][extra_classes]
                extra_labels = labels[extra_classes]
                indexes = [maxiou_per_gt, extra_anchors]
                maxiou_per_gt = np.concatenate(indexes,
                                            axis=0)
                labels = np.concatenate([labels, extra_labels],
                                        axis=0)

        # mask generation
        gt_mask = np.zeros((iou.shape[0], 4))
        # only indexes maxiou_per_gt are valid bounding boxes
        gt_mask[maxiou_per_gt] = 1.0

        # class generation
        gt_class = np.zeros((iou.shape[0], n_classes))
        # by default all are background (index 0)
        gt_class[:, 0] = 1
        # but those that belong to maxiou_per_gt are not
        gt_class[maxiou_per_gt, 0] = 0
        # we have to find those column indexes (classes)
        maxiou_col = np.reshape(maxiou_per_gt,
                                (maxiou_per_gt.shape[0], 1))
        label_col = np.reshape(labels[:,4],
                            (labels.shape[0], 1)).astype(int)
        row_col = np.append(maxiou_col, label_col, axis=1)
        # the label of object in maxio_per_gt
        gt_class[row_col[:,0], row_col[:,1]]  = 1.0
        
        # offsets generation
        gt_offset = np.zeros((iou.shape[0], 4))

        #(cx, cy, w, h) format
        if normalize:
            anchors = self.minmax2centroid(anchors)
            labels = self.minmax2centroid(labels)
            # bbox = bounding box
            # ((bbox xcenter - anchor box xcenter)/anchor box width)/.1
            # ((bbox ycenter - anchor box ycenter)/anchor box height)/.1
            # Equation 11.4.8
            offsets1 = labels[:, 0:2] - anchors[maxiou_per_gt, 0:2]
            offsets1 /= anchors[maxiou_per_gt, 2:4]
            offsets1 /= 0.1

            # log(bbox width / anchor box width) / 0.2
            # log(bbox height / anchor box height) / 0.2
            # Equation 11.4.8 
            offsets2 = np.log(labels[:, 2:4]/anchors[maxiou_per_gt, 2:4])
            offsets2 /= 0.2  

            offsets = np.concatenate([offsets1, offsets2], axis=-1)

        # (xmin, xmax, ymin, ymax) format
        else:
            offsets = labels[:, 0:4] - anchors[maxiou_per_gt]

        gt_offset[maxiou_per_gt] = offsets

        return gt_class, gt_offset, gt_mask

    def nms(self, classes, offsets, anchors, class_threshold=0.5, soft_nms=False, iou_threshold=0.2):
        """Perform NMS (Algorithm 11.12.1).
        Arguments:
            args : User-defined configurations
            classes (tensor): Predicted classes
            offsets (tensor): Predicted offsets
            
        Returns:
            objects (tensor): class predictions per anchor
            indexes (tensor): indexes of detected objects
                filtered by NMS
            scores (tensor): array of detected objects scores
                filtered by NMS
        """

        # get all non-zero (non-background) objects
        objects = np.argmax(classes, axis=1)
        # non-zero indexes are not background
        nonbg = np.nonzero(objects)[0]

        indexes = []
        while True:
            # list of zero probability values
            scores = np.zeros((classes.shape[0],))
            # set probability values of non-background
            scores[nonbg] = np.amax(classes[nonbg], axis=1)

            # max probability given the list
            score_idx = np.argmax(scores, axis=0)
            score_max = scores[score_idx]
            
            # get all non max probability & set it as new nonbg
            nonbg = nonbg[nonbg != score_idx]

            # if max obj probability is less than threshold
            if score_max < class_threshold:
                # we are done
                break

            indexes.append(score_idx)
            score_anc = anchors[score_idx]
            score_off = offsets[score_idx][0:4]
            score_box = score_anc + score_off
            score_box = np.expand_dims(score_box, axis=0)
            nonbg_copy = np.copy(nonbg)

            # perform Non-Max Suppression (NMS)
            for idx in nonbg_copy:
                anchor = anchors[idx]
                offset = offsets[idx][0:4]
                box = anchor + offset
                box = np.expand_dims(box, axis=0)
                iou = self.iou(box, score_box)[0][0]
                # if soft NMS is chosen
                if soft_nms:
                    # adjust score
                    iou = -2 * iou * iou
                    classes[idx] *= math.exp(iou)
                # else NMS (iou threshold def 0.2)
                elif iou >= iou_threshold:
                    # remove overlapping predictions with iou>threshold
                    nonbg = nonbg[nonbg != idx]

            # nothing else to process
            if nonbg.size == 0:
                break


        # get the array of object scores
        scores = np.zeros((classes.shape[0],))
        scores[indexes] = np.amax(classes[indexes], axis=1)

        return objects, indexes, scores