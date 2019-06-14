import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import RandomMaclaurin


def polynomial(X, Y, degree, bias=0):
    return (safe_sparse_dot(X, Y.T, True)+bias)**degree


def exp_kernel(X, Y, gamma):
    return np.exp(safe_sparse_dot(X, Y.T) * gamma)


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)


def test_random_maclaurin_polynomial():
    # compute exact kernel
    for degree in range(2, 5):
        kernel = polynomial(X, Y, degree)
        # approximate kernel mapping
        rm_transform = RandomMaclaurin(n_components=100, degree=degree,
                                       random_state=rng, kernel='poly')
        X_trans = rm_transform.fit_transform(X)
        Y_trans = rm_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)
        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.001)
        assert_less_equal(np.max(error), 0.01)  # nothing too far off
        assert_less_equal(np.mean(error), 0.005)  # mean is fairly close


def test_random_maclaurin_polynomial_bias():
    # compute exact kernel
    for bias in [0.01, 0.1, 1]:
        for degree in range(2, 5):
            print('bias: {} degree: {}'.format(bias, degree))
            kernel = polynomial(X, Y, degree, bias=bias)
            # approximate kernel mapping
            rm_transform = RandomMaclaurin(n_components=4000, degree=degree,
                                           random_state=rng, kernel='poly',
                                           bias=bias)
            X_trans = rm_transform.fit_transform(X)
            Y_trans = rm_transform.transform(Y)
            kernel_approx = np.dot(X_trans, Y_trans.T)
            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.01)
            assert_less_equal(np.max(error), 0.1)  # nothing too far off
            assert_less_equal(np.mean(error), 0.05)  # mean is fairly close


def test_random_maclaurin_polynomial_bias_h01():
    # compute exact kernel
    for bias in [0.01, 0.1, 1]:

        for degree in range(2, 5):
            kernel = polynomial(X, Y, degree, bias=bias)
            # approximate kernel mapping
            print('bias: {} degree: {}'.format(bias, degree))
            rm_transform = RandomMaclaurin(n_components=4000, degree=degree,
                                           random_state=rng, kernel='poly',
                                           bias=bias, h01=True)
            X_trans = rm_transform.fit_transform(X)
            Y_trans = rm_transform.transform(Y)
            kernel_approx = np.dot(X_trans, Y_trans.T)
            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.01)
            assert_less_equal(np.max(error), 0.1)  # nothing too far off
            assert_less_equal(np.mean(error), 0.05)  # mean is fairly close


def test_random_maclaurin_exp():
    # compute exact kernel
    kernel = exp_kernel(X, Y, 0.1)
    # approximate kernel mapping
    rm_transform = RandomMaclaurin(n_components=6000,
                                   random_state=rng, kernel='exp',
                                   gamma=0.1)

    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close