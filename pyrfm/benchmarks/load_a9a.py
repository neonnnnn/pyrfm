from sklearn.datasets import fetch_mldata


def load_data(datadir='datasets/'):
    """ Loads the dataset

    :type datadir: string
    :param datadir: the path to the dataset (here a9a)
    """

    #############
    # LOAD DATA #
    #############

    a9a = fetch_mldata('a9a', data_name=datadir)
    X_train, y_train = a9a.data[:32561], a9a.target[:32561]
    X_test, y_test  = a9a.data[32561:], a9a.target[32561:]
    return X_train, y_train, X_test, y_test
