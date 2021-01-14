#!/usr/bin/env python3
"""
Defines function that trains a loaded neural network model
using mini-batch gradient descent
"""


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    parameters:
        X_train [numpy.ndarray of shape (m, 784)]:
            contains training data
            m: number of data points
            784: the number of input features
        Y_train [one-hot numpy.ndarray of shape (m, 10)]:
            contains training labels
            m: number of data points, same as in X
            10: number of classes the model should classify
        X_valid [numpy.ndarray of shape (m, 784)]:
            contains validation data
            m: number of data points
            784: the number of input features
        Y_valid [one-hot numpy.ndarray of shape (m, 10)]:
            contains validation labels
            m: number of data points, same as in X
            10: number of classes the model should classify
        batch_size [int]:
            number of data points in a batch
        epochs [int]:
            number of times the training should pass through the whole dataset
        load_path [str]:
            path from which to load the neural network model
        save_path [str]:
            path to where the neural network should be saved after training

    loaded model will have the following tensors / ops in collection:
        x: placeholder for the input data
        y: placeholder for the labels
        accuracy: op to calculate the accuracy of the model
        loss: op to calculate the cost of the model
        train_op: op to perform one pass of gradient descent on the model
    before each epoch, training data should be shuffled
    print training and validation statements before each epoch
    print step, cost, and accuracy information every 100 steps within an epoch

    returns:
        the path where the model was saved
    """
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        print(x)
        print(y)
        print(accuracy)
        print(loss)
        print(train_op)

        # for epoch in range(epochs):
            # X_train, Y_train = shuffle_data(X_train, Y_train)
            # print("After {} epochs:".format(epoch))
            # print("\tTraining Cost: {}".format(train_cost))
            # print("\tTraining Accuracy: {}".format(train_accuracy))
            # print("\tValidation Cost: {}".format(valid_cost))
            # print("\tValidation Accuracy: {}".format(valid_accuracy))
    return (save_path)
