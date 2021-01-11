#!/usr/bin/env python3
"""
Defines a function that evaluates output of
neural network classifier
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates output of neural network

    parameters:
        X [numpy.ndarray]: contains the input data to evaluate
        Y [numpy.ndarray]: contains the one-hot labels for X
        save_path [string]: location to load the model from

    returns:
        the network's prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
    return (prediction, accuracy, loss)
