import numpy as np

from sklearn.utils.testing import (assert_greater_equal, assert_almost_equal,
                                   assert_less_equal)
from pyrfm import (TensorSketch, RandomKernel, RandomMaclaurin, RandomFourier,
                   AdaGradClassifier, AdaGradRegressor)
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
        # overfitting
        clf = AdaGradRegressor(transform, max_iter=100, warm_start=True,
                               verbose=False, fit_intercept=True, loss=loss,
                               alpha=0.0001, random_state=0,
                               normalize=normalize)
        clf.fit(X_train, y_train)
        l2 = np.mean((y_train - clf.predict(X_train)) ** 2)
        assert_less_equal(l2, 0.01)

        # underfitting
        clf_under = AdaGradRegressor(transform, max_iter=100, warm_start=True,
                                     verbose=False, fit_intercept=True,
                                     loss=loss, alpha=100000, random_state=0,
                                     normalize=normalize)
        clf_under.fit(X_train, y_train)
        assert_greater_equal(np.sum(clf.coef_ ** 2),
                             np.sum(clf_under.coef_ ** 2))

        # l1 regularization
        clf_l1 = AdaGradRegressor(transform, max_iter=100, warm_start=True,
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
        test_l2_sgd = np.mean((y_test - sgd.predict(X_trans[n_train:])) ** 2)
        clf = AdaGradRegressor(transform, max_iter=100, warm_start=True,
                               verbose=False, fit_intercept=True, loss=loss,
                               alpha=0.01, random_state=0, normalize=normalize)
        clf.fit(X_train, y_train)
        test_l2 = np.mean((y_test - clf.predict(X_test)) ** 2)
        assert_less_equal(test_l2, test_l2_sgd)

        # fast solver and slow solver
        clf_slow = AdaGradRegressor(transform, max_iter=10, warm_start=True,
                                    verbose=False, fit_intercept=True,
                                    loss=loss,
                                    alpha=0.0001, random_state=0,
                                    normalize=normalize, fast_solver=False)
        clf_slow.fit(X_train[:20], y_train[:20])

        clf_fast = AdaGradRegressor(transform, max_iter=10, warm_start=True,
                                    verbose=False, fit_intercept=True,
                                    loss=loss,
                                    alpha=0.0001, random_state=0,
                                    normalize=normalize, fast_solver=True)
        clf_fast.fit(X_train[:20], y_train[:20])
        assert_almost_equal(clf_fast.coef_, clf_slow.coef_, decimal=6)


def _test_classifier(transform, y_train, y_test, X_trans, normalize=False):
    for loss in ['squared_hinge', 'hinge', 'log']:
        # overfitting
        clf = AdaGradClassifier(transform, max_iter=500, warm_start=True,
                                verbose=False, fit_intercept=True, loss=loss,
                                alpha=0.0001, random_state=0,
                                normalize=normalize)
        clf.fit(X_train[:100], y_train[:100])
        train_acc = clf.score(X_train[:100], y_train[:100])
        assert_almost_equal(train_acc, 1)

        # underfitting
        clf_under = AdaGradClassifier(transform, max_iter=100, warm_start=True,
                                      verbose=False, fit_intercept=True,
                                      loss=loss, alpha=10000000000,
                                      random_state=0, normalize=normalize)
        clf_under.fit(X_train, y_train)
        assert_greater_equal(np.sum(clf.coef_ ** 2),
                             np.sum(clf_under.coef_ ** 2))

        # l1 regularization
        clf_l1 = AdaGradClassifier(transform, max_iter=100, warm_start=True,
                                   verbose=False, fit_intercept=True,
                                   loss=loss, alpha=10000000000, l1_ratio=1.,
                                   random_state=0, normalize=normalize)
        clf_l1.fit(X_train, y_train)
        assert_almost_equal(np.sum(np.abs(clf_l1.coef_)), 0)

        # comparison with sgd
        sgd = SGDClassifier(alpha=0.01, max_iter=100, eta0=1,
                            learning_rate='constant', fit_intercept=True,
                            loss=loss, random_state=0)
        sgd.fit(X_trans[:n_train], y_train)
        test_acc_sgd = sgd.score(X_trans[n_train:], y_test)
        clf = AdaGradClassifier(transform, max_iter=100, warm_start=True,
                                verbose=False, fit_intercept=True, loss=loss,
                                alpha=0.01, random_state=0, normalize=normalize)

        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        assert_greater_equal(test_acc, np.maximum(0.8, test_acc_sgd))

        # fast solver and slow solver
        clf_slow = AdaGradClassifier(transform, max_iter=10, warm_start=True,
                                     verbose=False, fit_intercept=True,
                                     loss=loss,
                                     alpha=0.0001, random_state=0,
                                     normalize=normalize, fast_solver=False)
        clf_slow.fit(X_train[:20], y_train[:20])

        clf_fast = AdaGradClassifier(transform, max_iter=10, warm_start=True,
                                     verbose=False, fit_intercept=True,
                                     loss=loss,
                                     alpha=0.0001, random_state=0,
                                     normalize=normalize, fast_solver=True)
        clf_fast.fit(X_train[:20], y_train[:20])
        assert_almost_equal(clf_fast.coef_, clf_slow.coef_, decimal=6)


def test_adagrad_regressor_ts():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_regressor(transform, y_train, y_test, X_trans)


def test_adagrad_regressor_ts_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)


def test_adagrad_regressor_rk():
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


def test_adagrad_regressor_rk_normalize():
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


def test_adagrad_regressor_rk_as():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans)


def test_adagrad_regressor_rk_as_normalize():
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


def test_adagrad_regressor_rm():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans)


def test_adagrad_regressor_rm_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)


def test_adagrad_regressor_rf():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, False)


def test_adagrad_regressor_rf_normalize():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    X_trans = StandardScaler().fit_transform(X_trans)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_regressor(transform, y_train, y_test, X_trans, True)


def test_adagrad_regressor_warm_start():
    rng = np.random.RandomState(0)
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]

    clf = AdaGradRegressor(transform, max_iter=10, warm_start=True,
                           verbose=False, fit_intercept=False, alpha=0.0001,
                           random_state=0, shuffle=False)
    clf.fit(X_train, y_train)

    clf_warm = AdaGradRegressor(transform, max_iter=2, warm_start=True,
                                verbose=False, fit_intercept=False,
                                alpha=0.0001, random_state=0,
                                shuffle=False)
    for i in range(5):
        clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf_warm.coef_, clf.coef_, decimal=3)


def test_adagrad_classifier_ts():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]
    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_adagrad_classifier_ts_normalize():
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


def test_adagrad_classifier_rk():
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


def test_adagrad_classifier_rk_normalize():
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


def test_adagrad_classifier_rk_as():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomKernel(n_components=100, random_state=0,
                             kernel='all_subsets')
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_adagrad_classifier_rk_as_normalize():
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


def test_adagrad_classifier_rm():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomMaclaurin(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_adagrad_classifier_rm_normalize():
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


def test_adagrad_classifier_rf():
    rng = np.random.RandomState(0)
    # approximate kernel mapping
    transform = RandomFourier(n_components=100, random_state=0, gamma=10)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = y[:n_train]
    y_test = y[n_train:]

    _test_classifier(transform, np.sign(y_train), np.sign(y_test), X_trans)


def test_adagrad_classifier_rf_normalize():
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


def test_adagrad_classifier_warm_start():
    rng = np.random.RandomState(0)
    transform = TensorSketch(n_components=100, random_state=0)
    X_trans = transform.fit_transform(X)
    y, coef = generate_target(X_trans, rng, -0.1, 0.1)
    y_train = np.sign(y[:n_train])

    clf = AdaGradClassifier(transform, max_iter=10, warm_start=True,
                            verbose=False, fit_intercept=False, alpha=0.0001,
                            random_state=0, shuffle=False)
    clf.fit(X_train, y_train)

    clf_warm = AdaGradClassifier(transform, max_iter=2, warm_start=True,
                                 verbose=False, fit_intercept=False,
                                 alpha=0.0001, random_state=0,
                                 shuffle=False)
    for i in range(5):
        clf_warm.fit(X_train, y_train)
    assert_almost_equal(clf_warm.coef_, clf.coef_, decimal=3)
