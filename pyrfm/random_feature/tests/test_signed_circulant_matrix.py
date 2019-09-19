import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.testing import (assert_less_equal, assert_allclose,
                                   assert_allclose_dense_sparse)
from pyrfm import SignedCirculantRandomMatrix
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import circulant

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


def test_signed_circulant_matrix():
    for gamma, n_components in zip([0.1, 1, 10], [10000, 5000, 2500]):
        # compute exact kernel
        kernel = rbf_kernel(X, Y, gamma)
        # approximate kernel mapping
        transformer = SignedCirculantRandomMatrix(n_components=n_components, 
                                                  gamma=gamma,
                                                  random_state=0)
        X_trans = transformer.fit_transform(X)
        Y_trans = transformer.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.01)
        assert_less_equal(np.max(error), 0.1)  # nothing too far off
        assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
        # for sparse matrix
        X_trans_sp = transformer.transform(csr_matrix(X))
        assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_signed_circulant_random_matrix_for_dot():
    # compute exact kernel
    kernel = np.dot(X, Y.T)
    # approximate kernel mapping
    n_components = X.shape[1]
    transformer = SignedCirculantRandomMatrix(n_components=n_components,
                                              random_fourier=False,
                                              random_state=0)
    X_trans = transformer.fit_transform(X)
    Y_trans = transformer.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.abs(np.mean(error)), 0.01)
    assert_less_equal(np.max(error), 0.1)  # nothing too far off
    assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
    # for sparse matrix
    X_trans_sp = transformer.transform(csr_matrix(X))
    assert_allclose_dense_sparse(X_trans, X_trans_sp)

    # comparing naive computation
    circ = circulant(transformer.random_weights_[0])
    circ *= transformer.random_sign_.T
    X_trans_naive = np.dot(X, circ.T) / np.sqrt(n_components)
    assert_allclose(X_trans, X_trans_naive)