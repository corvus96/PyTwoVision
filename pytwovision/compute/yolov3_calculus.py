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

    def bbox_giou(self, boxes1, boxes2):
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        # Calculate the iou value between the two bounding boxes
        iou = inter_area / union_area

        # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # Calculate the area of the smallest closed convex surface C
        enclose_area = enclose[..., 0] * enclose[..., 1]

        # Calculate the GIoU value according to the GioU formula  
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_ciou(self, boxes1, boxes2):
        boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
        up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
        right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
        down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

        c = (right - left) * (right - left) + (up - down) * (up - down)
        iou = self.bbox_iou(boxes1, boxes2)

        u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
        d = u / c

        ar_gt = boxes2[..., 2] / boxes2[..., 3]
        ar_pred = boxes1[..., 2] / boxes1[..., 3]

        ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
        alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
        ciou_term = d + alpha * ar_loss

        return iou - ciou_term
    
    def loss(self, pred, conv, label, bboxes, num_class, i=0, strides=[8, 16, 32], loss_thresh=0.5):
        strides = np.array(strides)

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = strides[i] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_class))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < loss_thresh, tf.float32 )

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        # Calculate the loss of confidence
        # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss
    
    def best_bboxes_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def nms(self, bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)
        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0 
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest score according to socre order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
                iou = self.best_bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
        valid_scale=[0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)