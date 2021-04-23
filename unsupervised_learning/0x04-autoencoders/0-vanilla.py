#!/usr/bin/env python3
"""
Defines function that creates a vanilla autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a "vanilla" autoencoder

    parameters:
        input_dims [int]:
            contains the dimensions of the model input
        hidden_layers [list of ints]:
            contains the number of nodes for each hidden layer in the encoder
                the hidden layers should be reversed for the decoder
        latent_dims [int]:
            contains the dimensions of the latent space representation

    All layers should use relu activation except for last layer
    Last layer should use sigmoid activation
    Autoencoder model should be compiled with Adam optimization
        and binary cross-entropy loss

    returns:
        encoder, decoder, auto
            encoder [model]: the encoder model
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

    encoder_inputs = keras.Input(shape=(input_dims,))
    encoder_value = encoder_inputs
    for i in range(0, len(hidden_layers)):
        encoder_layer = keras.layers.Dense(units=hidden_layers[i],
                                           activation='relu')
        encoder_value = encoder_layer(encoder_value)
    encoder_latent_layer = keras.layers.Dense(units=latent_dims,
                                              activation='relu')
    encoder_outputs = encoder_latent_layer(encoder_value)
    encoder = keras.Model(inputs=encoder_inputs, outputs=encoder_outputs)

    return encoder, None, None
