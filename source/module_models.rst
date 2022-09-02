=======
Tensorflow models
=======
It is in charge of generating the layers and blocks that make up the architectures of the different neural networks using the tools offered by Tensorflow. Within models there are two submodules blocks and layers, the first one stores groups of layers, which can be Tensorflow's own or customized layers which are found in the layers submodule. 

Blocks
-------
    .. automodule:: py2vision.models.blocks.backbone_block
        :members:

Layers
-------
    .. automodule:: py2vision.models.layers.batch_normalization_layer
        :members:

    .. automodule:: py2vision.models.layers.conv2d_bn_leaky_relu_layer
        :members:

    .. automodule:: py2vision.models.layers.upsample_layer
        :members:

    .. automodule:: py2vision.models.layers.residual_layer
        :members:

Models manager
--------------
Since the number of existing network architectures for recognition is enormous, the role of the model manager is to manage the use of network architectures by means of a mediator class.

    .. automodule:: py2vision.models.models_manager
        :members:

    .. automodule:: py2vision.models.yolov3_model
        :members:

    .. automodule:: py2vision.models.yolov3_tiny_model
        :members:

How to use?
^^^^^^^^^^^

.. code-block:: python
    :linenos:

    from py2vision.models.models_manager import ModelManager
    from py2vision.models.blocks.backbone_block import BackboneBlock
    from py2vision.models.blocks.backbone_block import darknet53

    np.random.seed(2000)
    input_data = np.random.randint(0, 255, size=(416, 416, 3))

    backbone_net = BackboneBlock(darknet53())
    model_manager = ModelManager()
    conv_sbbox, conv_mbbox, conv_lbbox, _ = model_manager.build_yolov3(backbone_net, 80)(self.input_data.shape)