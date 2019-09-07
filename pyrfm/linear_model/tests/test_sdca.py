import numpy as np

from sklearn.utils.testing import (assert_greater_equal, assert_almost_equal,
                                   assert_less_equal)
from pyrfm import (TensorSketch, RandomKernel, RandomMaclaurin, RandomFourier,
                   SDCAClassifier, SDCARegressor)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from .utils_linear_model import generate_target, generate_samples

# generate data
n_samples = 600
n_train = 500
n_features = 10
X = generate_samples(n_samples, n_features, 0)
X_train = X[:n_train]
X_test = X[n_train:]


def _test_regressor(transform, y_train, y_test, X_trans, normalize=False):
    for loss in ['squared']:
        # fast solver and slow solver
        clf_slow = SDCARegressor(transform, max_iter=10, warm_start=True,
                                 verbose=False, fit_intercept=True, loss=loss,
                                 alpha=0.0001, random_state=0,
                                 normalize=normalize, fast_solver=False)
        clf_slow.fit(X_train, y_train)

        clf_fast = SDCARegressor(transform, max_iter=10, warm_start=True,
                                 verbose=False, fit_intercept=True, loss=loss,
                                 alpha=0.0001, random_state=0,
                                 normalize=normalize, fast_solver=True)
        clf_fast.fit(X_train, y_train)
        assert_almost_equal(clf_fast.coef_, clf_slow.coef_, decimal=6)


        # overfitting
        clf = SDCARegressor(transform, max_iter=100, warm_start=True,
                            verbose=False, fit_intercept=True, loss=loss,
                            alpha=0.0001, random_state=0,
                            normalize=normalize)
        clf.fit(X_train, y_train)
        l2 = np.mean((y_train - clf.predict(X_train))**2)
        assert_less_equal(l2, 0.01)
        
        # underfitting
        clf_under = SDCARegressor(transform, max_iter=100, warm_start=True,
                                  verbose=False, fit_intercept=True, loss=loss,
                                  alpha=100000, random_state=0,
                                  normalize=normalize)
        clf_under.fit(X_train, y_train)
        assert_greater_equal(np.sum(clf.coef_ ** 2),
                             np.sum(clf_under.coef_ ** 2))

        # l1 regularization
        clf_l1 = SDCARegressor(transform, max_iter=100, warm_start=True,
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
        clf = SDCARegressor(transform, max_iter=100, warm_start=True,
                            verbose=False, fit_intercept=True, loss=loss,
                            alpha=0.01, random_state=0, normalize=normalize)
        clf.fit(X_train, y_train)
        test_l2 = np.mean((y_test - clf.predict(X_test))**2)
        assert_less_equal(test_l2, test_l2_sgd)


def _test_classifier(transform, y_train, y_test, X_trans, normalize=False):
    for loss in ['squared_hinge', 'log']:
        # overfitting
        clf = SDCAClassifier(transform, max_iter=100, warm_start=True,
                             verbose=False, fit_intercept=True, loss=loss,
                             alpha=0.0001, random_state=0,
                             normalize=normalize)
        clf.fit(X_train[:10], y_train[:10])
        train_acc = clf.score(X_train[:10], y_train[:10])
        assert_almost_equal(train_acc, 1)

        # underfitting
        clf_under = SDCAClassifier(transform, max_iter=100, warm_start=True,
                                   verbose=False, fit_intercept=True,
                                   loss=loss, alpha=10000,
                                   random_state=0, normalize=normalize)
        clf_under.fit(X_train, y_train)
        assert_greater_equal(np.sum(clf.coef_ ** 2),
                             np.sum(clf_under.coef_ ** 2))

        # l1 regularization
        clf_l1 = SDCAClassifier(transform, max_iter=100, warm_start=True,
                                verbose=False, fit_intercept=True,
                                loss=loss, alpha=1000, l1_ratio=0.9,
                                random_state=0, normalize=normalize)
        clf_l1.fit(X_train, y_train)
        assert_almost_equal(np.sum(np.abs(clf_l1.coef_)), 0)

        # comparison with sgd
        sgd = SGDClassifier(alpha=0.01, max_iter=100, eta0=1,
                            learning_rate='constant', fit_intercept=True,
                            loss=loss, random_state=0)
        sgd.fit(X_trans[:n_train], y_train)
        test_acc_sgd = sgd.score(X_trans[n_train:], y_test)
        clf = SDCAClassifier(transform, max_iter=100, warm_start=True,
                             verbose=False, fit_intercept=True, loss=loss,
                             alpha=0.01, random_state=0, normalize=normalize)

        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        assert_greater_equal(test_acc, np.maximum(0.8, test_acc_sgd))

        # fast solver and slow solver
        clf_slow = SDCAClassifier(transform, max_iter=10, warm_start=True,
                                  verbose=False, fit_intercept=True, loss=loss,
                                  alpha=0.0001, random_state=0,
                                  normalize=normalize, fast_solver=False)
        clf_slow.fit(X_train[:20], y_train[:20])

        clf_fast = SDCAClassifier(transform, max_iter=10, warm_start=True,
                                  verbose=False, fit_intercept=True, loss=loss,
                                  alpha=0.0001, random_state=0,
                                  normalize=normalize, fast_solver=True)
        clf_fast.fit(X_train[:20], y_train[:20])
        assert_almost_equal(clf_fast.coef_, clf_slow.coef_, decimal=6)


def test_sdca_classifier_ts():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_sdca_regressor_ts():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans)


def test_sdca_classifier_ts_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     True)


def test_sdca_regressor_ts_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)


def test_sdca_classifier_rk():
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        # approximate kernel mapping
        transform = RandomKernel(n_components=100, random_state=0,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        y, coef = generate_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]

        _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_sdca_regressor_rk():
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        # approximate kernel mapping
        transform = RandomKernel(n_components=100, random_state=0,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        y, coef = generate_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]

        _test_regressor(transform, y_train, y_test, X_trans)


def test_sdca_classifier_rk_normalize():
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        # approximate kernel mapping
        transform = RandomKernel(n_components=100, random_state=0,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        X_trans = StandardScaler().fit_transform(X_trans)

        y, coef = generate_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]

        _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                         True)


def test_sdca_regressor_rk_normalize():
    rng = np.random.RandomState(0)
    for degree in range(2, 5):
        # approximate kernel mapping
        transform = RandomKernel(n_components=100, random_state=0,
                                 degree=degree)
        X_trans = transform.fit_transform(X)
        X_trans = StandardScaler().fit_transform(X_trans)

        y, coef = generate_target(X_trans, rng, -0.1, 0.1)
        y_train = y[:n_train]
        y_test = y[n_train:]
        _test_regressor(transform, y_train, y_test, X_trans, True)


def test_sdca_classifier_rk_as():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_sdca_regressor_rk_as():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans)


def test_sdca_classifier_rk_as_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     True)


def test_sdca_regressor_rk_as_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)


def test_sdca_classifier_rm():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_sdca_regressor_rm():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans)


def test_sdca_classifier_rm_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     True)


def test_sdca_regressor_rm_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)


def test_sdca_classifier_rf():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_sdca_regressor_rf():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans)


def test_sdca_classifier_rf_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans,
                     True)


def test_sdca_regressor_rf_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    print(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)
