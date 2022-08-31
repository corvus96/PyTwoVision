import cv2 as cv
import numpy as np

from pytwovision.image_process.frame_decorator import FrameDecorator

class ResizeWithBBox(FrameDecorator):
    def apply(self, target_size, gt_boxes=None):
        """Apply resizing on an image and their bounding boxes.
        
        Args:
             target_size: a tuple or list with the new dimensions of an image.
             gt_boxes: bounding boxes. 
        
        Returns:
            a resized image and their reescaling bounding boxes.
        """
        ih, iw    = target_size
        h,  w, _  = self._frame.apply().shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv.resize(self._frame.apply(), (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes