#!/usr/bin/env python3
"""
Defines function that creates a convolutional autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    parameters:
        input_dims [tuple of ints]:
            contains the dimensions of the model input
        filters [list of ints]:
            contains the number of filters for each convolutional layer
                in the encoder
                the filters should be reversed for the decoder
        latent_dims [tuple of int]:
            contains the dimensions of the latent space representation

    Each convolution in the encoder should use kernel size of (3, 3) with
        same padding and relu activation,
        followed by max pooling of size (2, 2)
    Each convolution in the decoder should use filter size of (3, 3) with
        same padding and relu activation,
        followed by upsampling of size (2, 2),
        except last two:
    Second to last convolution should instead use valid padding
    Last convolution should have the same number of filters as the number of
        channels in input_dims with sigmoid activation and no upsampling
    Autoencoder model should be compiled with Adam optimization
        and binary cross-entropy loss

    returns:
        encoder, decoder, auto
            encoder [model]: the encoder model
            decoder [model]: the decoder model
            auto [model]: full autoencoder model
                compiled with adam optimization and binary cross-entropy loss
    """
    if type(input_dims) is not tuple:
        raise TypeError(
            "input_dims must be tuple of ints containing dimensions of \
            model input")
    for dim in input_dims:
        if type(dim) is not int:
            raise TypeError("input_dims must be tuple of ints containing \
            dimensions of model input")
    if type(filters) is not list:
        raise TypeError("filters must be a list of ints \
        representing number of filters for each convolutional layer")
    for number_filter in filters:
        if type(number_filter) is not int:
            raise TypeError("filters must be a list of ints \
            representing number of filters for each convolutional layer")
    if type(latent_dims) is not tuple:
        raise TypeError("latent_dims must be an int containing \
        dimensions of latent space representation")
    for dim in latent_dims:
        if type(dim) is not int:
            raise TypeError("latent_dims must be an int containing \
            dimensions of latent space representation")
    return None, None, None
