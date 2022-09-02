=======
Image process
=======

The concept of this module is based on being able to apply different effects to input images using a common class and if necessary stack one effect on top of another to that image.

The epicenter of this module
-----------------------------

    .. automodule:: py2vision.image_process.frame_decorator
        :members:

Image transformations
----------------------
    .. automodule:: py2vision.image_process.resize_with_bbox
        :members:
    .. automodule:: py2vision.image_process.resize
        :members:
    .. automodule:: py2vision.image_process.rotate
        :members:
    .. automodule:: py2vision.image_process.split_pair
        :members:

How to use?
-----------
.. code-block:: python
    :linenos:
    
    from py2vision.image_process.frame_decorator import Frame
    from py2vision.image_process.resize import Resize

    width = 640
    height = 720
    img = cv.imread("AN_IMAGE_PATH")
    frame = Frame(img)
    frame_transformed = Resize(frame).apply(width, height)
