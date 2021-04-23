#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    parameters:
        input_dims [int]:
            contains the dimensions of the model input
        hidden_layers [list of ints]:
            contains the number of nodes for each hidden layer in the encoder
                the hidden layers should be reversed for the decoder
        latent_dims [int]:
            contains the dimensions of the latent space representation

    All layers should use relu activation except for the mean and log
        variance layers in the encoder, which should use None,
        and the last layer, which should use sigmoid activation
    Autoencoder model should be compiled with Adam optimization
        and binary cross-entropy loss

    returns:
        encoder, decoder, auto
            encoder [model]: the encoder model,
                which should output the latent representation, the mean,
                and the log variance
            decoder [model]: the decoder model
            auto [model]: full autoencoder model
                compiled with adam optimization and binary cross-entropy loss
    """
    if type(input_dims) is not int:
        raise TypeError(
            "input_dims must be an int containing dimensions of model input")
    if type(hidden_layers) is not list:
        raise TypeError("hidden_layers must be a list of ints \
        representing number of nodes for each layer")
    for nodes in hidden_layers:
        if type(nodes) is not int:
            raise TypeError("hidden_layers must be a list of ints \
            representing number of nodes for each layer")
    if type(latent_dims) is not int:
        raise TypeError("latent_dims must be an int containing dimensions of \
        latent space representation")
    return None, None, None
