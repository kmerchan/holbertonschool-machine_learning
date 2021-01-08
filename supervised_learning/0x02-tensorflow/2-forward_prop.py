#!/usr/bin/env python3
"""
Defines a function that creates the forward propagation graph
for the neural network
"""


import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    parameters:
        x [tf.placeholder]: placeholder for input data
        layer_size [list]: contains number of nodes in each layer of network
        activations [list]: contains activation functions for each layer

    returns:
        prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i is 0:
            output = create_layer(x, layer_sizes[i], activations[i])
        else:
            output = create_layer(output, layer_sizes[i], activations[i])
    return output
