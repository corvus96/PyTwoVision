=======
Image process
=======

The concept of this module is based on being able to apply different effects to input images using a common class and if necessary stack one effect on top of another to that image.

The epicenter of this module
-----------------------------

    .. automodule:: pytwovision.image_process.frame_decorator
        :members:

Image transformations
----------------------
    .. automodule:: pytwovision.image_process.resize_with_bbox
        :members:
    .. automodule:: pytwovision.image_process.resize
        :members:
    .. automodule:: pytwovision.image_process.rotate
        :members:
    .. automodule:: pytwovision.image_process.split_pair
        :members:

How to use
-----------
.. code-block:: python
    :linenos:
    
    from pytwovision.image_process.frame_decorator import Frame
    from pytwovision.image_process.resize import Resize

    width = 640
    height = 720
    img = cv.imread("AN_IMAGE_PATH")
    frame = Frame(img)
    frame_transformed = Resize(frame).apply(width, height)
