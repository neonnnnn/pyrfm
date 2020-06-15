import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import TensorSketch
import pytest


def polynomial(X, Y, degree):
    return safe_sparse_dot(X, Y.T, dense_output=True)**degree


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


@pytest.mark.parametrize("degree", [2,3,4])
def test_tensor_sketching(degree):
    # compute exact kernel
    kernel = polynomial(X, Y, degree)
    # approximate kernel mapping
    ts_transform = TensorSketch(n_components=1000, degree=degree,
                                random_state=rng)
    X_trans = ts_transform.fit_transform(X)
    Y_trans = ts_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) < 0.001
    assert np.max(error) < 0.01  # nothing too far off
    assert np.mean(error) < 0.005  # mean is fairly close

    X_trans_sp = ts_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)
