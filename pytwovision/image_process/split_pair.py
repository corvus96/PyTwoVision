from pytwovision.image_process.frame_decorator import FrameDecorator

class SplitPair(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, mode="sbs"):
        """Apply split in an image pair from stereo cameras.

        Args:
            mode: if it is "sbs" (side-by-side) it will do a vertical slice, if it is "tb" (top-bottom) it will do a a horizontal slice.
            
        Returns: 
            left image and right image.
        """
        # dividing height and width by 2
        height, width = self._frame.apply().shape[:2]
        new_width = int(width/2)
        new_height = int(height/2)
        if mode == "sbs":
            img_left = self._frame.apply()[0:height,0:new_width] #Y+H and X+W
            img_right = self._frame.apply()[0:height,new_width:width]
        elif mode == "tb":
            img_left = self._frame.apply()[0:new_height, 0:width] #Y+H and X+W
            img_right = self._frame.apply()[new_height:height, 0:width]
        else:
            ValueError("mode just can be 'sbs' or 'tb'")
        return img_left, img_right