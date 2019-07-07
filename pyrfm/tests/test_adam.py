import numpy as np
from sklearn.utils.testing import assert_greater_equal, assert_almost_equal
from pyrfm import (MB, TensorSketch, RandomKernel, RandomMaclaurin,
                   SignedCirculantRandomKernel, RandomFourier, AdamClassifier)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# generate data
X = np.random.RandomState(0).random_sample(size=(500, 10)) * 2 - 1
n_train = int(X.shape[0] * 0.9)
n_test = X.shape[0] - n_train
X_train = X[:n_train]
X_test = X[n_train:]


def make_target(X_trans, rng, low=-1., high=1.0, ):
    coef = rng.uniform(low, high, size=X_trans.shape[1])
    y = np.dot(X_trans, coef)
    y -= np.mean(y)
    y = np.sign(y)
    return y, coef


def test_adam_classifier_ts():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=rng)
    X_trans = transform.fit_transform(X)
    y, coef = make_target(X_trans, rng, -1, 1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_ts_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=rng)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = make_target(X_trans, rng, -1, 1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rk():
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        # approximate kernel mapping
        transform = RandomKernel(n_components=100, random_state=rng,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        y, coef = make_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]

        clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                                random_state=rng, l1_ratio=0,
                                alpha=1, max_iter=100,
                                fit_intercept=True)
        clf_sgd.fit(X_trans[:n_train], y_train)
        test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
        train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
        clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                             verbose=False, fit_intercept=True,
                             alpha=1 / X.shape[0], random_state=rng)
        clf.fit(X_train, y_train)

        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        assert_greater_equal(test_acc, test_acc_sgd)
        assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rk_normalize():
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        # approximate kernel mapping
        transform = RandomKernel(n_components=100, random_state=rng,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        X_trans = StandardScaler().fit_transform(X_trans)

        y, coef = make_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]

        clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                                random_state=rng, l1_ratio=0,
                                alpha=1, max_iter=100,
                                fit_intercept=True)
        clf_sgd.fit(X_trans[:n_train], y_train)
        test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
        train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
        clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                             verbose=False, fit_intercept=True,
                             alpha=1 / X.shape[0], random_state=rng)
        clf.fit(X_train, y_train)

        test_acc = clf.score(X_test, y_test)
        train_acc = clf.score(X_train, y_train)
        assert_greater_equal(test_acc, test_acc_sgd)
        assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rk_as():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=rng,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rk_as_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=rng,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rm():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=rng)
    X_trans = transform.fit_transform(X)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rm_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=rng)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rf():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=rng, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_rf_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=rng, gamma=10)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2',
                            random_state=rng, l1_ratio=0,
                            alpha=1, max_iter=100,
                            fit_intercept=True)
    clf_sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = clf_sgd.score(X_trans[n_train:], y_test)
    train_acc_sgd = clf_sgd.score(X_trans[:n_train], y_train)
    clf = AdamClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True,
                         alpha=1 / X.shape[0], random_state=rng)
    clf.fit(X_train, y_train)

    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    assert_greater_equal(test_acc, test_acc_sgd)
    assert_greater_equal(train_acc, train_acc_sgd)


def test_adam_classifier_warm_start():
    rng = np.random.RandomState(0)
    transform = TensorSketch(n_components=100, random_state=rng)
    X_trans = transform.fit_transform(X)
    y, coef = make_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]

    clf = AdamClassifier(transform, max_iter=10, warm_start=True,
                         verbose=False, fit_intercept=False, alpha=1,
                         random_state=rng,
                         shuffle=False)
    clf.fit(X_train, y_train)

    clf_warm = AdamClassifier(transform, max_iter=2, warm_start=True,
                              verbose=False, fit_intercept=False, alpha=1,
                              random_state=rng,
                              shuffle=False)
    for i in range(5):
        clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf_warm.coef_, clf.coef_, decimal=3)
