import numpy as np
from sklearn.datasets import fetch_mldata


def load_data(datadir='datasets/'):
    """
    Loads the dataset

    :type datadir: string
    :param datadir: the path to the dataset (here MNIST)
    """

    mnist = fetch_mldata('MNIST original', data_name=datadir)
    rng = np.random.RandomState(1)
    idx = rng.permutation(mnist.data.shape[0])
    data = mnist.data[idx].astype(np.float64)
    target = mnist.target[idx]
    data /= np.max(data)
    X_train, y_train = data[:50000], target[:50000]
    X_test, y_test  = data[60000:], target[60000:]
    return X_train, y_train, X_test, y_test
