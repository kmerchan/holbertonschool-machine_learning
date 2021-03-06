#!/usr/bin/env python3
"""
Defines class Yolo that uses the Yolo v3 algorithm to perform object detection
"""


import tensorflow.keras as K
import numpy as np
import cv2
import glob


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
        def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
            suppresses non-max filter boxes to return predicted bounding box
        def preprocess_images(self, images):
            resizes and rescales the images before processeing

    static methods:
        def sigmoid(x):
            passes x through sigmoid function, so output is between 0 & 1
        def load_images(folder_path):
            loads images
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
        return None

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
        return None

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Suppresses all non-max filter boxes to return predicted bounding box

        parameters:
            filtered boxes [numpy.ndarray of shape (?, 4)]:
                contains all filtered bounding boxes
            box_classes [numpy.ndarray of shape (?,)]:
                contains the class number that each box
                    in filtered boxes predicts
            box_score [numpy.ndarray of shape (?)]:
                contains the box scores for each box
                    in filtered boxes

        returns:
            tuple of (box_predictions, predicted_box_classes,
                        predicted_box_scores):
                box_predictions [numpy.ndarray of shape (?, 4)]:
                    contains all predicted bounding boxes
                predicted_box_classes [numpy.ndarray of shape (?,)]:
                    contains the class number that each box
                        in box predictions
                predicted_box_score [numpy.ndarray of shape (?)]:
                    contains the box scores for each box
                        in box predictions
        """
        return None

    @staticmethod
    def load_images(folder_path):
        """
        Loads images

        parameters:
            folder_path [str]: path to the folder holding all images to load

        returns:
            tuple of (images, image_paths):
                images [list]: images as numpy.ndarrays
                image_paths [list]: paths to the individual images
        """
        image_paths = glob.glob(folder_path + "/*")
        images = []
        for image in image_paths:
            images.append(cv2.imread(image))
        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Resizes and rescales the images before processeing

        parameters:
            images [list]: images as numpy.ndarrays

        Resizes the images with inter-cubic interpolation
        Rescales the images to have pixel values in the range [0, 1]

        returns:
            tuple of (pimages, image_shapes):
                pimages [numpy.ndarray of shape (ni, input_h, input_w, 3):
                    contatins all preprocessed images
                    ni: number of images preprocessed
                    input_h: input height for Darknet model
                    input_w: input width for Darknet model
                    3: number of color channels
                image_shapes [numpy.ndarray of shape (n1, 2)]:
                    contains the original height and width of image
                    ni: number of images preprocessed
                    2: (image_height, image_width)
        """
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value

        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(image, dsize=(input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255
            pimages.append(rescaled)
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)
