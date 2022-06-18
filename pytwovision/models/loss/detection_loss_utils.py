import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber



def focal_loss_binary(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Formula:
        loss = - alpha*sum(alpha*((1-pt)^gamma)*log(pt))
    Args:
        y_pred: An array or tensor with predicted score 
        y_true: An array or tensor with objetive value
        alpha: balancing factor, default value is 0.25.
        gamma: modulating factor, default value is 2.0.
    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
          shape as `y_true`; otherwise, it is scalar.
    """

    pt_1 = tf.where(tf.equal(y_true, 1),
                    y_pred,
                    tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0),
                    y_pred,
                    tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN and Inf
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    weight = alpha * K.pow(1. - pt_1, gamma)
    fl1 = -K.sum(weight * K.log(pt_1))
    weight = (1 - alpha) * K.pow(pt_0, gamma)
    fl0 = -K.sum(weight * K.log(1. - pt_0))

    return fl1 + fl0


def focal_loss_categorical(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Categorical cross-entropy focal loss"""

    # scale to ensure sum of prob is 1.0
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # clip the prediction value to prevent NaN and Inf
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # calculate cross entropy
    cross_entropy = -y_true * K.log(y_pred)

    # calculate focal loss
    weight = alpha * K.pow(1 - y_pred, gamma)
    cross_entropy *= weight

    return K.sum(cross_entropy, axis=-1)


def mask_offset(y_true, y_pred): 
    """Pre-process ground truth and prediction data"""
    # 1st 4 are offsets
    offset = y_true[..., 0:4]
    # last 4 are mask
    mask = y_true[..., 4:8]
    # pred is actually duplicated for alignment
    # either we get the 1st or last 4 offset pred
    # and apply the mask
    pred = y_pred[..., 0:4]
    offset *= mask
    pred *= mask
    return offset, pred


def l1_loss(y_true, y_pred):
    """MAE or L1 loss
    """
    offset, pred = mask_offset(y_true, y_pred)
    # we can use L1
    return K.mean(K.abs(pred - offset), axis=-1)


def smooth_l1_loss(y_true, y_pred):
    """Smooth L1 loss using tensorflow Huber loss
    """
    offset, pred = mask_offset(y_true, y_pred)
    # Huber loss as approx of smooth L1
    return Huber()(offset, pred)

