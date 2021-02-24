#!/usr/bin/env python3
"""
Defines class Yolo that uses the Yolo v3 algorithm to perform object detection
"""


import tensorflow.keras as K


class Yolo:
    """
    Class that uses Yolo v3 algorithm to perform object detection

    class constructor:
        def __init__(self, model_path, classes_path, class_t, nms_t, anchors)

    public instance attributes:
        model: the Darknet Keras model
        class_names: list of all the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo class constructor

        parameters:
            model_path [str]: the path to where a Darknet Keras model is stored
            classes_path [str]: the path to where the list of class names
                used for the Darknet model can be found,
                list is ordered by order of index
            class_t [float]: represents the box score threshold for
                the initial filtering step
            nms_t [float]: represents the IOU threshold for non-max suppression
            anchors [numpy.ndarray of shape (outputs, anchor_boxes, 2)]:
                contains all the anchor boxes:
                outputs: the number of predictions made by the Darknet model
                anchor_boxes: number of anchor boxes used for each prediction
                2: [anchor_box_width, anchor_box_height]
        """
        self.model = K.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
