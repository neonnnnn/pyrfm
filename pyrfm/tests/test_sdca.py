import numpy as np


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from pyrfm import (MB, TensorSketch, RandomKernel, RandomMaclaurin,
                   SignedCirculantRandomKernel, RandomFourier,
                   SDCAClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(500, 50))*2-1
#X /= np.sum(X, axis=1, keepdims=True)*2-1
n_train = int(X.shape[0]*0.9)
n_test = X.shape[0] - n_train


def test_sdca_classifier_ts():
    # approximate kernel mapping
    transform = TensorSketch(n_components=1000, random_state=rng)
    X_trans = transform.fit_transform(X)
    coef = rng.normal(0.2, 4, size=(X_trans.shape[1]))
    y = np.dot(X_trans, coef)+0.5
    y = np.sign(y)
    y_train = y[:n_train]
    y_test = y[n_train:]
    X_train = X[:n_train]
    X_test = X[n_train:]
    clf_baseline = LogisticRegression(C=1, fit_intercept=True, max_iter=1000)
    clf_baseline.fit(X_trans[:n_train], y_train)

    clf = SDCAClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True, alpha=1,
                         random_state=1, tol=0, loss='logistic')
    clf.fit(X_train, y_train)
    test_acc_base = clf_baseline.score(X_trans[n_train:], y_test)
    test_acc_ada = clf.score(X_test, y_test)
    assert_almost_equal(test_acc_ada-test_acc_base, 0, decimal=1)
    assert_almost_equal(clf_baseline.score(X_trans[:n_train], y_train),
                        clf.score(X_train, y_train))


def test_sdca_classifier_ts_normalize():
    # approximate kernel mapping
    transform = TensorSketch(n_components=1000, random_state=rng)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)

    coef = rng.normal(0.2, 4, size=(X_trans.shape[1]))
    y = np.dot(X_trans, coef)+0.5
    y = np.sign(y)
    y_train = y[:n_train]
    y_test = y[n_train:]
    X_train = X[:n_train]
    X_test = X[n_train:]
    clf_baseline = LogisticRegression(C=1, fit_intercept=False, max_iter=1000)
    clf_baseline.fit(X_trans[:n_train], y_train)

    clf = SDCAClassifier(transform, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=False, alpha=1,
                         normalize=True, random_state=1, loss='logistic')
    clf.fit(X_train, y_train)

    test_acc_base = clf_baseline.score(X_trans[n_train:], y_test)
    test_acc_ada = clf.score(X_test, y_test)
    assert_almost_equal(test_acc_ada-test_acc_base, 0, decimal=1)
    assert_almost_equal(clf_baseline.score(X_trans[:n_train], y_train),
                        clf.score(X_train, y_train))
