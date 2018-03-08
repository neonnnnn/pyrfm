import numpy
import os
from sklearn.datasets import fetch_mldata


def load_data(datadir='datasets/'):
    ''' Loads the dataset

    :type datadir: string
    :param datadir: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    mnist = fetch_mldata('MNIST_original', data_name=datadir)
    X_train, y_train = mnist.data[:50000], mnist.target[:50000]
    X_test, y_test  = mnist.data[60000:], mnist.target[60000:]
    return X_train, y_train, X_test, y_test
