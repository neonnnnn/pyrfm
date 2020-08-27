import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import RandomMaclaurin
import pytest
import itertools


def polynomial(X, Y, degree, bias=0):
    return (safe_sparse_dot(X, Y.T, dense_output=True)+bias)**degree


def exp_kernel(X, Y, gamma):
    return np.exp(safe_sparse_dot(X, Y.T) * gamma)


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


params = itertools.product([0, 0.01, 0.1, 1], [2,3,4], [True, False])


@pytest.mark.parametrize("bias,degree,h01", params)
def test_random_maclaurin_polynomial_bias(bias, degree, h01):
    # compute exact kernel
    print('bias: {} degree: {}'.format(bias, degree))
    kernel = polynomial(X, Y, degree, bias=bias)
    # approximate kernel mapping
    rm_transform = RandomMaclaurin(n_components=5000, degree=degree,
                                   random_state=rng, kernel='poly',
                                   bias=bias, h01=h01)
    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) < 0.01
    assert np.max(error) < 0.1  # nothing too far off
    assert np.mean(error) < 0.05  # mean is fairly close

    X_trans_sp = rm_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_random_maclaurin_exp():
    # compute exact kernel
    kernel = exp_kernel(X, Y, 0.1)
    # approximate kernel mapping
    rm_transform = RandomMaclaurin(n_components=6000, random_state=rng,
                                   kernel='exp', gamma=0.1)

    X_trans = rm_transform.fit_transform(X)
    Y_trans = rm_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) < 0.01
    assert np.max(error) < 0.1  # nothing too far off
    assert np.mean(error) < 0.05  # mean is fairly close

    X_trans_sp = rm_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)
