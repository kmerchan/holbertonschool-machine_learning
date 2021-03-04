#!/usr/bin/env python3
"""
Defines class Yolo that uses the Yolo v3 algorithm to perform object detection
"""


import tensorflow.keras as K
import numpy as np


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

    public methods:
        def process_outputs(self, outputs, image_size):
            calculates scaled coordinates of boundary boxes from outputs
        def filter_boxes(self, boxes, box_confidences, box_class_probs):
            returns all filtered bounding boxes from processed outputs
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
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            lines = f.readlines()
            self.class_names = []
            for name in lines:
                self.class_names.append(name[:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """
        Returns the output after passing through Sigmoid function
        output will be between 0 and 1
        """
        return (1. / (1. + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs

        parameters:
            outputs [list of numpy.ndarrays]:
                contains predictions from the Darknet model for a single image
            image_size [numpy.ndarray]:
                contains the image's original size [image_height, image_width]

        Each output has the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            grid_height: height of the grid used for the output
            grid_width: width of the grid used for the output
            anchor_boxes: number of anchor boxes used
            4: (t_x, t_y, t_w, t_h)
            1: box confidence
            classes: class probabilities for all classes

        returns:
            tuple of (boxes, box_confidences, box_class_probs):
                boxes [list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 4)]:
                    contains processed boundary boxes for each output:
                        4: (x1, y1, x2, y2)
                        (x1, y1, x2, y2) should represent the boundary box
                            relative to original image
                box_confidences [list of numpy.ndarray of shape
                    (grid_height, grid_width, anchor_boxes, 1)]:
                    contains box confidences for each output
                box_class_probs [list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, classes)]:
                    contains box's class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            grid_height, grid_width = output.shape[:2]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            sigmoid_conf = self.sigmoid(output[..., 4])
            sigmoid_prob = self.sigmoid(output[..., 5:])

            box_conf = np.expand_dims(sigmoid_conf, axis=-1)
            box_class_prob = sigmoid_prob

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.inputs[0].shape.as_list()[1:3]

            grid = np.tile(np.indices((grid_width, grid_height)).T,
                           anchors.shape[0]).reshape(
                               (grid_height, grid_width) + anchors.shape)

            b_xy = (self.sigmoid(t_xy) + grid) / [grid_width, grid_height]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Determines filtered bounding boxes from processed outputs

        parameters:
            boxes [list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)]:
                contains processed boundary boxes for each output:
                    4: (x1, y1, x2, y2)
                    (x1, y1, x2, y2) should represent the boundary box
                        relative to original image
            box_confidences [list of numpy.ndarray of shape
                (grid_height, grid_width, anchor_boxes, 1)]:
                contains box confidences for each output
            box_class_probs [list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)]:
                contains box's class probabilities for each output

        returns:
            tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes [numpy.ndarray of shape (?, 4)]:
                    contains all filtered bounding boxes
                box_classes [numpy.ndarray of shape (?,)]:
                    contains the class number that each box
                        in filtered boxes predicts
                box_scores [numpy.ndarray of shape (?)]:
                    contains the box scores for each box
                        in filtered boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, b in enumerate(boxes):
            box_conf = box_confidences[i]
            box_class_prob = box_class_probs[i]

            box_score = box_conf * box_class_prob

            box_class = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)

            index = np.where(box_class_score >= self.class_t)

            filtered_boxes.append(b[index])
            box_classes.append(box_class[index])
            box_scores.append(box_class_score[index])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)
        return (filtered_boxes, box_classes, box_scores)
