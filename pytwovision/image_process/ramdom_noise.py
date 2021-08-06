from frame_decorator import FrameDecorator, Frame

class RamdomNoise(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, input_image):
        """
        Decorators may call parent implementation of the operation, instead of
        calling the wrapped object directly. This approach simplifies extension
        of decorator classes.
        """
        return input_image + "Recoil 1"


class RamdomNoise2(FrameDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def apply(self, input_image):
        """
        Decorators may call parent implementation of the operation, instead of
        calling the wrapped object directly. This approach simplifies extension
        of decorator classes.
        """
        return input_image + "Recoil 2"

def client_code(frame: Frame):
    """
    The client code works with all objects using the Component interface. This
    way it can stay independent of the concrete classes of components it works
    with.
    """

    # ...

    print(frame.apply())


if __name__ == "__main__":
    # This way the client code can support both simple components...
    simple = Frame()
    recurso = simple.apply("gotcha")
    print(simple.apply("gotcha"))
    print("Client: I've got a simple component:")
    print("\n")
    decorator1 = RamdomNoise(simple)
    print(decorator1._frame)
    decorator2 = RamdomNoise2(decorator1)

    print(decorator1.apply(recurso))
    print(decorator2.apply("morrow"))