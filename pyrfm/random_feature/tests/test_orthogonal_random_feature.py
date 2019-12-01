import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.testing import (assert_less_equal,
                                   assert_allclose_dense_sparse)
from pyrfm import OrthogonalRandomFeature, StructuredOrthogonalRandomFeature
from sklearn.metrics.pairwise import rbf_kernel
import pytest


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


params = [
    (10, 2048, True), (10, 2048, False),
    (100, 4096, True), (100, 4096, False)
]


@pytest.mark.parametrize("gamma, n_components, use_offset", params)
def test_orthogonal_random_feature(gamma, n_components, use_offset):
    # compute exact kernel
    kernel = rbf_kernel(X, Y, gamma)
    # approximate kernel mapping
    rf_transform = OrthogonalRandomFeature(n_components=n_components,
                                           gamma=gamma, use_offset=use_offset,
                                           random_state=0)
    X_trans = rf_transform.fit_transform(X)
    Y_trans = rf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
    # for sparse matrix
    X_trans_sp = rf_transform.transform(csr_matrix(X))
    assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_orthogonal_random_feature_for_dot():
    # compute exact kernel
    kernel = np.dot(X, Y.T)
    # approximate kernel mapping
    rf_transform = OrthogonalRandomFeature(n_components=64,
                                           random_fourier=False,
                                           random_state=0)
    X_trans = rf_transform.fit_transform(X)
    Y_trans = rf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
    # for sparse matrix
    X_trans_sp = rf_transform.transform(csr_matrix(X))
    assert_allclose_dense_sparse(X_trans, X_trans_sp)



@pytest.mark.parametrize("gamma, n_components, use_offset", params)
def test_structured_orthogonal_random_feature(gamma, n_components, use_offset):
    # compute exact kernel
    kernel = rbf_kernel(X, Y, gamma)
    # approximate kernel mapping
    rf_transform = StructuredOrthogonalRandomFeature(
        n_components=n_components,
        use_offset=use_offset,
        gamma=gamma, random_state=0
    )
    X_trans = rf_transform.fit_transform(X)
    Y_trans = rf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
    # for sparse matrix
    X_trans_sp = rf_transform.transform(csr_matrix(X))
    assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_structured_orthogonal_random_feature_for_dot():
    # compute exact kernel
    kernel = np.dot(X, Y.T)
    # approximate kernel mapping
    rf_transform = StructuredOrthogonalRandomFeature(
        n_components=64, random_fourier=False,
        random_state=0
    )
    X_trans = rf_transform.fit_transform(X)
    Y_trans = rf_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
    # for sparse matrix
    X_trans_sp = rf_transform.transform(csr_matrix(X))
    assert_allclose_dense_sparse(X_trans, X_trans_sp)