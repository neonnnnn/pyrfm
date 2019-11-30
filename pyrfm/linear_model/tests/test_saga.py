import numpy as np

from sklearn.utils.testing import (assert_greater_equal, assert_almost_equal,
                                   assert_less_equal)
from pyrfm import (TensorSketch, RandomKernel, RandomMaclaurin, RandomFourier,
                   AdditiveChi2Sampler, SAGAClassifier, SAGARegressor)
from sklearn.kernel_approximation import (RBFSampler, SkewedChi2Sampler)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from .utils_linear_model import generate_target, generate_samples
from scipy.sparse import csr_matrix
import pytest
from itertools import product


loss_reg = ['squared']
loss_clf = ['squared_hinge', 'log', 'hinge']

# generate data
n_samples = 500
n_train = 400
n_features = 8
X = generate_samples(n_samples, n_features, 0)
X_train = X[:n_train]
X_test = X[n_train:]


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_regularization(normalize, loss):
    rng = np.random.RandomState(0)
    transformer = RBFSampler(n_components=100, random_state=0, gamma=10)
    transformer.fit(X)
    X_trans = transformer.transform(X)
    if normalize:
        X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    # overfitting
    clf = SAGARegressor(transformer, max_iter=300, warm_start=True,
                        verbose=False, fit_intercept=True, loss=loss,
                        alpha=0.0001, intercept_decay=1e-6,
                        random_state=0, tol=0, normalize=normalize)
    clf.fit(X_train[:100], y_train[:100])
    l2 = np.mean((y_train[:100] - clf.predict(X_train[:100]))**2)
    assert_less_equal(l2, 0.01)

    # underfitting
    clf_under = SAGARegressor(transformer, max_iter=100, warm_start=True,
                              verbose=False, fit_intercept=True, loss=loss,
                              alpha=100000, random_state=0,
                              normalize=normalize)
    clf_under.fit(X_train, y_train)
    assert_greater_equal(np.sum(clf.coef_ ** 2),
                         np.sum(clf_under.coef_ ** 2))

    # l1 regularization
    clf_l1 = SAGARegressor(transformer, max_iter=100, warm_start=True,
                           verbose=False, fit_intercept=True,
                           loss=loss, alpha=1000, l1_ratio=0.9,
                           random_state=0, normalize=normalize)
    clf_l1.fit(X_train, y_train)
    assert_almost_equal(np.sum(np.abs(clf_l1.coef_)), 0)

    # comparison with sgd
    sgd = SGDRegressor(alpha=0.01, max_iter=100, eta0=1,
                       learning_rate='constant', fit_intercept=True,
                       random_state=0)
    sgd.fit(X_trans[:n_train], y_train)
    test_l2_sgd = np.mean((y_test - sgd.predict(X_trans[n_train:]))**2)
    clf = SAGARegressor(transformer, max_iter=100, warm_start=True,
                        verbose=False, fit_intercept=True, loss=loss,
                        alpha=0.01, random_state=0, normalize=normalize,
                        )

    clf.fit(X_train, y_train)
    test_l2 = np.mean((y_test - clf.predict(X_test))**2)
    assert_less_equal(test_l2, test_l2_sgd)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_regularization(normalize, loss):
    rng = np.random.RandomState(0)
    transformer = RBFSampler(n_components=100, random_state=0, gamma=10)
    transformer.fit(X)
    X_trans = transformer.transform(X)
    if normalize:
        X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    y_train = np.sign(y_train)
    y_test = np.sign(y_test)
    # overfitting
    clf = SAGAClassifier(transformer, max_iter=500, warm_start=True,
                         verbose=False, fit_intercept=True, loss=loss,
                         alpha=0.00001, intercept_decay=1e-10,
                         random_state=0, tol=0, eta0=0.01,
                         normalize=normalize)
    clf.fit(X_train[:100], y_train[:100])
    train_acc = clf.score(X_train[:100], y_train[:100])
    assert_greater_equal(train_acc, 0.95)
    # underfitting
    clf_under = SAGAClassifier(transformer, max_iter=100, warm_start=True,
                               verbose=False, fit_intercept=True,
                               loss=loss, alpha=10000,
                               random_state=0, normalize=normalize)
    clf_under.fit(X_train, y_train)
    assert_greater_equal(np.sum(clf.coef_ ** 2),
                         np.sum(clf_under.coef_ ** 2))

    # l1 regularization
    clf_l1 = SAGAClassifier(transformer, max_iter=100, warm_start=True,
                            verbose=False, fit_intercept=True, loss=loss,
                            alpha=1000, l1_ratio=0.9,
                            random_state=0, normalize=normalize)
    clf_l1.fit(X_train, y_train)
    assert_almost_equal(np.sum(np.abs(clf_l1.coef_)), 0)

    # comparison with SGD
    sgd = SGDClassifier(alpha=0.01, max_iter=100, eta0=1,
                        learning_rate='constant', fit_intercept=True,
                        loss=loss, random_state=0)
    sgd.fit(X_trans[:n_train], y_train)
    test_acc_sgd = sgd.score(X_trans[n_train:], y_test)
    clf = SAGAClassifier(transformer, max_iter=100, warm_start=True,
                         verbose=False, fit_intercept=True, loss=loss,
                         alpha=0.01, random_state=0, normalize=normalize,
                         eta0=0.1)

    clf.fit(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    assert_greater_equal(test_acc, test_acc_sgd)


def _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=False, sparse=True, loss='squared'):
    # fast solver and slow solver
    clf_slow = SAGARegressor(transformer, max_iter=10, warm_start=True,
                             verbose=False, fit_intercept=True, loss=loss,
                             alpha=0.0001, random_state=0,
                             normalize=normalize, fast_solver=False)
    clf_slow.fit(X_train, y_train)

    clf_fast = SAGARegressor(transformer, max_iter=10, warm_start=True,
                             verbose=False, fit_intercept=True, loss=loss,
                             alpha=0.0001, random_state=0,
                             normalize=normalize, fast_solver=True)
    clf_fast.fit(X_train, y_train)
    assert_almost_equal(clf_fast.coef_, clf_slow.coef_, decimal=6)
    if sparse:
        # dense / sparse
        clf_dense = SAGARegressor(transformer, loss=loss, max_iter=10,
                                  random_state=0, verbose=False)
        clf_dense.fit(X_train, y_train)
        clf_sparse = SAGARegressor(transformer, loss=loss, max_iter=10,
                                   random_state=0, verbose=False)
        clf_sparse.fit(csr_matrix(X_train), y_train)
        assert_almost_equal(clf_dense.coef_, clf_sparse.coef_)

    # warm_start
    clf = SAGARegressor(transformer, max_iter=10, shuffle=False,
                        random_state=0, tol=0, loss=loss,
                        warm_start=False, verbose=False)
    clf.fit(X_train, y_train)
    clf_warm = SAGARegressor(transformer, max_iter=5, shuffle=False,
                             random_state=0, tol=0, loss=loss,
                             warm_start=True, verbose=False)
    clf_warm.fit(X_train, y_train)
    clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf.t_, clf_warm.t_)
    assert_almost_equal(clf.dloss_, clf_warm.dloss_)
    assert_almost_equal(clf.coef_, clf_warm.coef_)


def _test_classifier(transformer, X_train, y_train, X_test, y_test, X_trans,
                     normalize=False, sparse=True, loss='squared_hinge'):
    # fast solver and slow solver
    clf_slow = SAGAClassifier(transformer, max_iter=10, warm_start=True,
                              verbose=False, fit_intercept=True, loss=loss,
                              random_state=0, normalize=normalize,
                              fast_solver=False)
    clf_slow.fit(X_train[:20], y_train[:20])

    clf_fast = SAGAClassifier(transformer, max_iter=10, warm_start=True,
                              verbose=False, fit_intercept=True, loss=loss,
                              random_state=0, normalize=normalize,
                              fast_solver=True)
    clf_fast.fit(X_train[:20], y_train[:20])
    assert_almost_equal(clf_fast.coef_, clf_slow.coef_, decimal=6)
    if sparse:
        # dense / sparse
        clf_dense = SAGAClassifier(transformer, loss=loss, max_iter=10,
                                   random_state=0, verbose=False)
        clf_dense.fit(X_train, y_train)
        clf_sparse = SAGAClassifier(transformer, loss=loss, max_iter=10,
                                    random_state=0, verbose=False)
        clf_sparse.fit(csr_matrix(X_train), y_train)
        assert_almost_equal(clf_dense.coef_, clf_sparse.coef_)

    # warm_start
    clf = SAGAClassifier(transformer, max_iter=10, shuffle=False,
                         random_state=0, tol=0, loss=loss,
                         warm_start=False, verbose=False)
    clf.fit(X_train, y_train)
    clf_warm = SAGAClassifier(transformer, max_iter=5, shuffle=False,
                              random_state=0, tol=0, loss=loss,
                              warm_start=True, verbose=False)
    clf_warm.fit(X_train, y_train)
    clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf.t_, clf_warm.t_)
    assert_almost_equal(clf.dloss_, clf_warm.dloss_)
    assert_almost_equal(clf.coef_, clf_warm.coef_)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_ts(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = TensorSketch(n_components=100, random_state=0)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transformer, X_train, np.sign(y_train), X_test,
                     np.sign(y_test), X_trans, normalize=normalize, 
                     loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_ts(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = TensorSketch(n_components=100, random_state=0)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=normalize, loss=loss)


@pytest.mark.parametrize("normalize, loss, degree", 
                         product([True, False], loss_clf, [2,3,4]))
def test_classifier_rk(normalize, loss, degree):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomKernel(n_components=100, random_state=0,
                                degree=degree)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transformer, X_train, np.sign(y_train), X_test,
                     np.sign(y_test), X_trans, normalize=normalize,
                     loss=loss)


@pytest.mark.parametrize("normalize, loss, degree", 
                         product([True, False], loss_reg, [2, 3, 4]))
def test_regressor_rk(normalize, loss, degree):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomKernel(n_components=100, random_state=0,
                                degree=degree)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=normalize, loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_rk_as(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomKernel(n_components=100, random_state=0,
                               kernel='all_subsets')
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(
        transformer,
        X_train,
        np.sign(y_train),
        X_test,
        np.sign(y_test),
        X_trans,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_rk_as(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomKernel(n_components=100, random_state=0,
                               kernel='all_subsets')
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=normalize, loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_rm(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(
        transformer,
        X_train,
        np.sign(y_train),
        X_test,
        np.sign(y_test),
        X_trans,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_rm(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=normalize, loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_rf(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(
        transformer,
        X_train,
        np.sign(y_train),
        X_test,
        np.sign(y_test),
        X_trans,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_rf(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=normalize, loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_rbf(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RBFSampler(n_components=100, random_state=0, gamma=10)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(
        transformer,
        X_train,
        np.sign(y_train),
        X_test,
        np.sign(y_test),
        X_trans,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_rbf(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = RBFSampler(n_components=100, random_state=0, gamma=10)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transformer, X_train, y_train, X_test, y_test, X_trans,
                    normalize=normalize, loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_skewed(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = SkewedChi2Sampler(n_components=100, random_state=0)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(
        transformer,
        X_train,
        np.sign(y_train),
        X_test,
        np.sign(y_test),
        X_trans,
        sparse=False,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_skewed(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = SkewedChi2Sampler(n_components=100, random_state=0)
    X_trans = transformer.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(
        transformer,
        X_train,
        y_train,
        X_test,
        y_test,
        X_trans,
        sparse=False,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_clf))
def test_classifier_additive(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = AdditiveChi2Sampler()
    X_trans = transformer.fit_transform(np.abs(X))
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(
        transformer,
        np.abs(X_train),
        np.sign(y_train),
        np.abs(X_test),
        np.sign(y_test),
        X_trans,
        normalize=normalize,
        loss=loss)


@pytest.mark.parametrize("normalize, loss", product([True, False], loss_reg))
def test_regressor_additive(normalize, loss):
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transformer = AdditiveChi2Sampler()
    X_trans = transformer.fit_transform(np.abs(X))
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(
        transformer,
        np.abs(X_train),
        y_train,
        np.abs(X_test),
        y_test,
        X_trans,
        normalize=normalize,
        loss=loss)
