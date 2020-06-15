import numpy as np

from sklearn.utils.testing import assert_almost_equal
from pyrfm import (TensorSketch, RandomKernel, RandomMaclaurin, RandomFourier,
                   DoublySGDClassifier, DoublySGDRegressor)
from sklearn.kernel_approximation import RBFSampler, SkewedChi2Sampler
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from .utils_linear_model import generate_target, generate_samples
import pytest
from itertools import product

loss_reg = ['squared']
loss_clf = ['squared_hinge', 'hinge', 'log']

# generate data
n_samples = 150
n_train = 120
n_features = 7
X = generate_samples(n_samples, n_features, 0)
X_train = X[:n_train]
X_test = X[n_train:]


def _test_regressor(transform, y_train, y_test, X_trans, max_iter=100, 
                    eta0=0.01, loss='squared'):
    # learn?
    clf = DoublySGDRegressor(transform, max_iter=max_iter, warm_start=True,
                                verbose=False, fit_intercept=True, loss=loss,
                                alpha=1e-7, intercept_decay=1e-7,
                                random_state=0, tol=0, power_t=1, eta0=eta0)
    clf.fit(X_train, y_train)
    l2 = np.mean((y_train - clf.predict(X_train))**2)
    assert l2 < 0.01
    
    # compare the norms of coefs: overfitting vs underfitting
    clf_over = DoublySGDRegressor(transform, warm_start=True,
                                    verbose=False, fit_intercept=True, loss=loss,
                                    alpha=1e-7, intercept_decay=1e-7,
                                    random_state=0, tol=0, power_t=1, eta0=eta0)
    clf_over.fit(X_train, y_train)
    # underfitting
    clf_under = DoublySGDRegressor(transform, warm_start=True,
                                    verbose=False, fit_intercept=True, loss=loss,
                                    alpha=10000, random_state=0, power_t=1,
                                    eta0=eta0)
    clf_under.fit(X_train, y_train)
    assert np.sum(clf_under.coef_ ** 2) < np.sum(clf_over.coef_ ** 2)

    # use same seed?
    assert_almost_equal(clf_over.predict(X_train),
                        clf_over.predict(X_train))
    
    # l1 regularization
    clf_l1 = DoublySGDRegressor(transform,
                                warm_start=True, verbose=False,
                                fit_intercept=True, loss=loss,
                                alpha=1000, l1_ratio=0.9, random_state=0,
                                power_t=1, eta0=eta0)
    clf_l1.fit(X_train, y_train)
    assert_almost_equal(np.sum(np.abs(clf_l1.coef_[:-2])), 0)

    # dense / sparse
    clf_dense = DoublySGDRegressor(transform, loss=loss, max_iter=10,
                                    random_state=0, eta0=eta0, verbose=False)
    clf_dense.fit(X_train, y_train)
    clf_sparse = DoublySGDRegressor(transform, loss=loss, max_iter=10,
                                    random_state=0, eta0=eta0, verbose=False)
    clf_sparse.fit(csr_matrix(X_train), y_train)
    assert_almost_equal(clf_dense.coef_, clf_sparse.coef_)

    # warm_start
    clf = DoublySGDRegressor(transform, max_iter=10, shuffle=False,
                                random_state=0, tol=0, loss=loss,
                                warm_start=False, verbose=False, eta0=eta0)
    clf.fit(X_train, y_train)
    clf_warm = DoublySGDRegressor(transform, max_iter=5, shuffle=False,
                                    random_state=0, tol=0, loss=loss,
                                    warm_start=True, verbose=False, eta0=eta0)
    clf_warm.fit(X_train, y_train)
    clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf.t_, clf_warm.t_)
    assert_almost_equal(clf.coef_[-1], clf_warm.coef_[-1])


def _test_classifier(transform, y_train, y_test, X_trans, max_iter=100,
                     eta0=0.01, loss='squared_hinge'):
    #  learn?
    clf = DoublySGDClassifier(transform, max_iter=max_iter,
                              verbose=True, fit_intercept=True, loss=loss,
                              alpha=1e-7, intercept_decay=1e-7, eta0=eta0,
                              random_state=0, tol=0, power_t=1)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    assert train_acc >= .80
        
    # compare the norms of coefs: overfitting vs underfitting
    clf_over = DoublySGDClassifier(transform, warm_start=True,
                                   verbose=False, fit_intercept=True,
                                   loss=loss, alpha=1e-7,
                                   intercept_decay=1e-7, eta0=eta0,
                                   random_state=0, tol=0, power_t=1)
    clf_over.fit(X_train, y_train)
    clf_under = DoublySGDClassifier(transform, warm_start=True,
                                    verbose=False, fit_intercept=True,
                                    loss=loss, alpha=1000, random_state=0,
                                    power_t=1, eta0=eta0)
    clf_under.fit(X_train, y_train)
    assert np.sum(clf_under.coef_ ** 2) < np.sum(clf_over.coef_ ** 2)

    # use same seed?
    assert_almost_equal(clf_over.decision_function(X_train),
                        clf_over.decision_function(X_train))
        
    # l1 regularization
    clf_l1 = DoublySGDClassifier(transform, verbose=False,
                                 fit_intercept=True,
                                 loss=loss, alpha=1000, l1_ratio=0.9,
                                 random_state=0, power_t=1, eta0=eta0)
    clf_l1.fit(X_train, y_train)
    assert_almost_equal(np.sum(np.abs(clf_l1.coef_[:-2])), 0)

    # dense / sparse
    clf_dense = DoublySGDClassifier(transform, loss=loss, max_iter=10,
                                    random_state=0, eta0=eta0)
    clf_dense.fit(X_train, y_train)
    clf_sparse = DoublySGDClassifier(transform, loss=loss, max_iter=10,
                                     random_state=0, eta0=eta0)
    clf_sparse.fit(csr_matrix(X_train), y_train)
    assert_almost_equal(clf_dense.coef_, clf_sparse.coef_)

    # warm_start
    clf = DoublySGDClassifier(transform, max_iter=10, shuffle=False,
                              random_state=0, tol=0, loss=loss, eta0=eta0)
    clf.fit(X_train, y_train)
    clf_warm = DoublySGDClassifier(transform, max_iter=5, shuffle=False,
                                   random_state=0, tol=0, loss=loss,
                                   warm_start=True, eta0=eta0)
    clf_warm.fit(X_train, y_train)
    clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf.t_, clf_warm.t_)
    assert_almost_equal(clf.coef_, clf_warm.coef_)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_rf(loss):
    rng = np.random.RandomState(0)
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     max_iter=500, loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_rf(loss):
    rng = np.random.RandomState(0)
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans, max_iter=100,
                    loss=loss)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_rf_use_offset(loss):
    rng = np.random.RandomState(0)
    transform = RandomFourier(n_components=100, random_state=0, gamma=10,
                              use_offset=True)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     max_iter=500, loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_rf_use_offset(loss):
    rng = np.random.RandomState(0)
    transform = RandomFourier(n_components=100, random_state=0, gamma=10,
                              use_offset=True)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans, loss=loss)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_rk(loss):
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        transform = RandomKernel(n_components=100, random_state=0,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        y, coef = generate_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]
        _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                         max_iter=100, eta0=.1, loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_rk(loss):
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        transform = RandomKernel(n_components=100, random_state=0,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        y, coef = generate_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]
        _test_regressor(transform, y_train, y_test, X_trans, loss=loss)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_rk_as(loss):
    rng = np.random.RandomState(0)
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_rk_as(loss):
    rng = np.random.RandomState(0)
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans, loss=loss)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_rm(loss):
    rng = np.random.RandomState(0)
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_rm(loss):
    rng = np.random.RandomState(0)
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans, loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_rbf(loss):
    rng = np.random.RandomState(0)
    transform = RBFSampler(n_components=100, gamma=10, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans, loss=loss)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_rbf(loss):
    rng = np.random.RandomState(0)
    transform = RBFSampler(n_components=100, gamma=10, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     max_iter=500, eta0=.01, loss=loss)


@pytest.mark.parametrize("loss", loss_reg)
def test_sgd_regressor_skewed(loss):
    rng = np.random.RandomState(0)
    transform = SkewedChi2Sampler(random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans, loss=loss)


@pytest.mark.parametrize("loss", loss_clf)
def test_sgd_classifier_skewed(loss):
    rng = np.random.RandomState(0)
    transform = SkewedChi2Sampler(random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     max_iter=300, loss=loss)
