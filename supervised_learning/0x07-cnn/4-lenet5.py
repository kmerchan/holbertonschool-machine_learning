#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using TensorFlow
"""


import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of LeNet-5 architecture using TensorFlow

    parameters:
        x [tf.placeholder of shape (m, 28, 28, 1)]:
            contains the input images for the network
            m: number of images
        y [tf.placeholder of shape (m, 10)]:
            contains the one-hot labels for the network

    model layers:
    C1: convolutional layer with 6 kernels of shape (5, 5) with same padding
    P2: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    C3: convolutional layer with 16 kernels of shape (5, 5) with valid padding
    P4: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    F5: fully connected layer with 120 nodes
    F6: fully connected layer with 84 nodes
    F7: fully connected softmax output layer with 10 nodes

    All layers requiring init should initialize kernels with he_normal method:
        tf.contrib.layers.variance_scaling_initializer()
    All hidden layer requiring activation should use relu activation function

    returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (default hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    weights_initializer = tf.contrib.layers.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size = (5, 5),
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=weights_initializer)
    output_1 = C1(x)
    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    output_2 = P2(output_1)
    C3 = tf.layers.Conv2D(filters=16,
                          kernel_size = (5, 5),
                          padding='valid',
                          activation=tf.nn.relu,
                          kernel_initializer=weights_initializer)
    output_3 = C3(output_2)
    P4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    output_4 = P4(output_3)
    output4 = t.layers.Flatten(output_4)
    F5 = tf.layers.Dense(
        120,
        activation=tf.nn.relu,
        kernel_initializer=weights_initializer)
    output_5 = F5(output_4)
    F6 = tf.layers.Dense(
        84,
        activation=tf.nn.relu,
        kernel_initializer=weights_initializer)
    output_6 = F6(output_5)
    F7 = tf.layers.Dense(
        10,
        activation=tf.nn.softmax,
        kernel_initializer=weights_initializer)
    output_7 = F7(output_6)
    loss = tf.losses.softmax_cross_entropy(output_7,
                                           logits=y)
    op = tf.train.AdamOptimizer().minimize(loss)
    y_pred =tf.math.argmax(y, axis=1)
    y_out = tf.math.argmax(output_7, axis=1)
    equality = tf.math.equality(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.case(equality, "float"))
    return output_7, op, loss, accuracy
