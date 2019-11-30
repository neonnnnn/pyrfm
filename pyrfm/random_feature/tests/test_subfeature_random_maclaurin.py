import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import SubfeatureRandomMaclaurin
import pytest
import itertools


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
X_sp = csr_matrix(X)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_subfeature_random_maclaurin_polynomial(degree):
    # compute exact kernel
    kernel = polynomial(X, Y, degree)
    # approximate kernel mapping
    rm_transform = SubfeatureRandomMaclaurin(n_components=500, degree=degree,
                                                random_state=rng, kernel='poly')
    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.001)
    assert_less_equal(np.max(error), 0.01)  # nothing too far off
    assert_less_equal(np.mean(error), 0.005)  # mean is fairly close

    X_trans_sp = rm_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)


@pytest.mark.parametrize("bias, degree", 
                         itertools.product([0.01, 0.1, 1], [2, 3, 4]))
def test_subfeature_random_maclaurin_polynomial_bias(bias, degree):
    # compute exact kernel
    print('bias: {} degree: {}'.format(bias, degree))
    kernel = polynomial(X, Y, degree, bias=bias)
    # approximate kernel mapping
    rm_transform = SubfeatureRandomMaclaurin(n_components=10000, degree=degree,
                                                n_sub_features=10,
                                                random_state=rng, kernel='poly',
                                                bias=bias)
    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close

    X_trans_sp = rm_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)


@pytest.mark.parametrize("bias, degree", 
                         itertools.product([0.01, 0.1, 1], [2, 3, 4]))
def test_subfeature_random_maclaurin_polynomial_bias_h01(bias, degree):
    # compute exact kernel
    kernel = polynomial(X, Y, degree, bias=bias)
    # approximate kernel mapping
    print('bias: {} degree: {}'.format(bias, degree))
    rm_transform = SubfeatureRandomMaclaurin(n_components=5000, degree=degree,
                                             n_sub_features=10,
                                             random_state=rng, kernel='poly',
                                             bias=bias, h01=True)
    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close

    X_trans_sp = rm_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_subfeature_random_maclaurin_exp():
    # compute exact kernel
    kernel = exp_kernel(X, Y, 0.1)
    # approximate kernel mapping
    rm_transform = SubfeatureRandomMaclaurin(n_components=10000, n_sub_features=10,
                                             random_state=rng, kernel='exp',
                                             gamma=0.1)

    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close

    X_trans_sp = rm_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)
