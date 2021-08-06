class Frame():
    """
    The base Component interface defines operations that can be altered by
    decorators.
    """
    def apply(self, input_image):
        return input_image

class FrameDecorator(Frame):
    """
    The base Decorator class follows the same interface as the other components.
    The primary purpose of this class is to define the wrapping interface for
    all concrete decorators. The default implementation of the wrapping code
    might include a field for storing a wrapped component and the means to
    initialize it.
    """

    _frame: Frame = None

    def __init__(self, frame: Frame):
        self._frame = Frame

    @property
    def frame(self):
        """
        The Decorator delegates all work to the wrapped component.
        """

        return self._frame

    def apply(self):
        
        return self._frame.apply()


