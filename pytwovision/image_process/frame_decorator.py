class Frame():
    """The base Component interface defines operations that can be altered by decorators.
    
    Attributes:
        img: an image array
    """
    def __init__(self, input_image):
        
        self.img = input_image
        
    def apply(self):
        """To get input image"""
        return self.img

class FrameDecorator(Frame):
    _frame: Frame = None

    def __init__(self, frame: Frame):
        """ The base Decorator class follows the same interface as the other components. The primary purpose of this class is to define the wrapping interface for all concrete decorators. The default implementation of the wrapping code might include a field for storing a wrapped component and the means to initialize it.
        
        Attributes:
            _frame: a Frame instance
        """
        if isinstance(frame, Frame):
            self._frame = frame
        else:
            self._frame = Frame(frame)

    @property
    def frame(self):
        """
        The Decorator delegates all work to the wrapped component.
        """

        return self._frame

    def apply(self, input_image):
        
        return self._frame.apply(input_image)


