#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer

    public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'

    instance attributes:
        style_image: preprocessed style image
        content_image: preprocessed style image
        alpha: weight for content cost
        beta: weight for style cost
        model: the Keras model used to calculate cost
        gram_style_features: list of gram matrices from style layer outputs
        content_feature: the content later output of the content image

    class constructor:
        def __init__(self, style_image, content_image, alpha=1e4, beta=1)

    static methods:
        def scale_image(image):
            rescales an image so the pixel values are between 0 and 1
                and the largest side is 512 pixels
        def gram_matrix(input_layer):
            calculates gram matrices

    public instance methods:
        def load_model(self):
            creates model used to calculate cost from VGG19 Keras base model
        def generate_features(self):
            extracts the features used to calculate neural style cost
        def layer_style_cost(self, style_output, gram_target):
            calculates the style cost for a single layer
        def style_cost(self, style_outputs):
            calculates the style cost for generated image
        def content_cost(self, content_output):
            calculates the content cost for the generated image
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class

        parameters:
            style_image [numpy.ndarray with shape (h, w, 3)]:
                image used as style reference
            content_image [numpy.ndarray with shape (h, w, 3)]:
                image used as content reference
            alpha [float]: weight for content cost
            beta [float]: weight for style cost

        Raises TypeError if input are in incorrect format
        Sets TensorFlow to execute eagerly
        Sets instance attributes
        """
        if type(style_image) is not np.ndarray or \
           len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or \
           len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
            and its largest side is 512 pixels

        parameters:
            image [numpy.ndarray of shape (h, w, 3)]:
                 image to be rescaled

        Scaled image should be tf.tensor with shape (1, h_new, w_new, 3)
            where max(h_new, w_new) is 512 and
            min(h_new, w_new) is scaled proportionately
        Image should be resized using bicubic interpolation.
        Image's pixels should be rescaled from range [0, 255] to [0, 1].

        returns:
            the scaled image
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(tf.expand_dims(image, 0),
                                          size=(h_new, w_new))
        rescaled = resized / 255
        rescaled = tf.div(tf.subtract(rescaled,
                                      tf.reduce_min(rescaled)),
                          tf.subtract(tf.reduce_max(rescaled),
                                      tf.reduce_min(rescaled)))
        return (rescaled)

    def load_model(self):
        """
        Creates the model used to calculate cost from VGG19 Keras base model

        Model's input should match VGG19 input
        Model's output should be a list containing outputs of VGG19 layers
            listed in style_layers followed by content_layers

        Saves the model in the instance attribute model
        """
        model = tf.keras.applications.VGG19(include_top=False,
                                            weights='imagenet')
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates gram matrices

        parameters:
            input_layer [an instance of tf.Tensor or tf.Variable
                of shape (1, h, w, c)]:
                contains the layer output to calculate gram matrix for

        returns:
            tf.Tensor of shape (1, c, c) containing gram matrix of input_layer
        """
        if not is_instance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) is not 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        return (input_layer)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost

        Sets public instance attribute:
            gram_style_features and content_feature
        """
        self.gram_style_features = []
        self.content_feature = []

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        parameters:
            style_output [tf.Tensor of shape (1, h, w, c)]:
                contains the layer style output of the generated image
            gram_target [tf.Tensor of shape (1, c, c)]:
                the gram matrix of the target style output for that layer

        returns:
            the layer's style cost
        """
        if not is_instance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) is not 4:
            raise TypeError("style_output must be a tensor of rank 4")
        one, h, w, c = style_output.shape
        if not is_instance(gram_target, (tf.Tensor, tf.Variable)) or \
           len(gram_target.shape) is not 3:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image

        parameters:
            style_outputs [list of tf.Tensors]:
                contains stye outputs for the generated image

        returns:
            the style cost
        """
        length = len(self.style_layers)
        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length))

    def content_cost(self, content_output):
        """
        Calculates the content cost for generated image

        parameters:
            content_output [tf.Tensor]:
                contains content output for the generated image

        returns:
            the style cost
        """
        shape = self.content_feature.shape
        if not is_instance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(shape))