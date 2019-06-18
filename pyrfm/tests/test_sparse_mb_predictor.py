import numpy as np


from sklearn.utils.testing import assert_less_equal
from pyrfm import MB, SparseMB
from pyrfm import SparseMBClassifier, SparseMBRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)


def test_mb_classifier():
    # approximate kernel mapping
    mb_transform = MB(n_components=1000)
    X_trans = mb_transform.fit_transform(X)
    coef = rng.normal(-0.4, 2, size=(X_trans.shape[1]))
    y = np.dot(X_trans, coef)
    y = np.sign(y)
    y_train = y[:250]
    y_test = y[250:]
    X_train = X[:250]
    X_test = X[250:]
    clf_lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000)
    clf_lr.fit(X_trans[:250], y_train)
    score_lr = clf_lr.score(X_trans[250:], y_test)

    clf = SparseMBClassifier(n_components=1000, max_iter=100, warm_start=True,
                             verbose=False, fit_intercept=False, alpha=0.1)
    clf.fit(X_train, y_train)
    assert_less_equal(score_lr, clf.score(X_test, y_test))


def test_mb_regressor():
    # approximate kernel mapping
    mb_transform = MB(n_components=1000)
    X_trans = mb_transform.fit_transform(X)
    coef = rng.normal(-0.4, 2, size=(X_trans.shape[1]))
    y = np.dot(X_trans, coef)
    y_train = y[:250]
    y_test = y[250:]
    X_train = X[:250]
    X_test = X[250:]
    clf_ridge = Ridge(fit_intercept=False)
    clf_ridge.fit(X_trans[:250], y_train)
    score_lr = clf_ridge.score(X_trans[250:], y_test)

    clf = SparseMBRegressor(n_components=1000, max_iter=100, warm_start=True,
                            verbose=False, fit_intercept=False, alpha=0.1)
    clf.fit(X_train, y_train)
    assert_less_equal(score_lr, clf.score(X_test, y_test))
