import numpy as np
from sklearn.utils.testing import assert_less_equal
from pyrfm import MB, SparseMB
from pyrfm import SparseMBClassifier, SparseMBRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from .utils_linear_model import generate_target, generate_samples

# generate data
n_samples = 500
n_train = 400
n_features = 50
X = generate_samples(n_samples, n_features, 0)
X = (X+1)/2.
X /= np.sum(X, axis=1, keepdims=True)
X_train = X[:n_train]
X_test = X[n_train:]


def test_mb_classifier():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    mb_transform = MB(n_components=1000)
    X_trans = mb_transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng)
    y = np.sign(y)
    y_train = y[:n_train]
    y_test = y[n_train:]

    clf_lr = LogisticRegression(C=10.0, fit_intercept=False, max_iter=1000)
    clf_lr.fit(X_trans[:n_train], y_train)
    score_lr = clf_lr.score(X_trans[n_train:], y_test)

    clf = SparseMBClassifier(n_components=1000, max_iter=100, warm_start=True,
                             verbose=False, fit_intercept=False, alpha=0.1)
    clf.fit(X_train, y_train)
    assert_less_equal(score_lr, clf.score(X_test, y_test))


def test_mb_regressor():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    mb_transform = MB(n_components=1000)
    X_trans = mb_transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng)

    y_train = y[:n_train]
    y_test = y[n_train:]
    clf_ridge = Ridge(fit_intercept=False)
    clf_ridge.fit(X_trans[:n_train], y_train)
    score_lr = clf_ridge.score(X_trans[n_train:], y_test)

    clf = SparseMBRegressor(n_components=1000, max_iter=100, warm_start=True,
                            verbose=False, fit_intercept=False, alpha=0.1)
    clf.fit(X_train, y_train)
    assert_less_equal(score_lr, clf.score(X_test, y_test))
