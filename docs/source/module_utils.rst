=======
Utilities
=======

Here you will find all those functionalities that do not fit in the rest of the modules.

Annotations Parser
-------------------
    .. automodule:: pytwovision.utils.annotations_parser
        :members:

Example of code
^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat

    anno_out_file = "test_anno_file"
    xml_path = "tests/test_dataset/annotations"
    classes_out_file = "test_classes"
    work_dir = "tests"
    images_path = "tests/test_dataset/images"

    parser = XmlParser()
    anno_format = YoloV3AnnotationsFormat()
    parser.parse(anno_format, xml_path, anno_out_file, classes_out_file, images_path, work_dir)

Annotations Helper
-------------------
    .. automodule:: pytwovision.utils.annotations_helper
        :members:

Example of code
^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    from pytwovision.utils.annotations_parser import XmlParser, YoloV3AnnotationsFormat
    from pytwovision.utils.annotations_helper import AnnotationsHelper

    anno_out_file = "annotations_formated"
    xml_path = "tests/test_dataset/annotations"
    classes_file = "test_dataset_generator"
    work_dir = "tests/test_dataset/to_generator_test"
    images_path = "tests/test_dataset/images"

    try:
        os.mkdir(work_dir)
    except:
        pass

    #create annotations formated
    parser = XmlParser()
    anno_format = YoloV3AnnotationsFormat()
    parser.parse(anno_format, xml_path, anno_out_file, classes_file, images_path, work_dir)

    training_percen = 0.8
    anno_out_full_path = os.path.join(work_dir, "{}.txt".format(anno_out_file))
    anno_helper = AnnotationsHelper(anno_out_full_path)

    train, test = anno_helper.split(training_percen)
    anno_helper.export(train, os.path.join(work_dir, "train.txt"))
    anno_helper.export(test, os.path.join(work_dir, "test.txt"))

Draw functions
-------------------
    .. automodule:: pytwovision.utils.draw 
        :members:

Label utils
-------------------
    .. automodule:: pytwovision.utils.label_utils
        :members: