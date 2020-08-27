import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.testing import assert_allclose
from pyrfm import CountSketch


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


def test_count_sketch():
    # compute exact kernel
    kernel = np.dot(X, Y.T)
    # approximate kernel mapping
    transformer = CountSketch(n_components=30, random_state=1)
    X_trans = transformer.fit_transform(X)
    Y_trans = transformer.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert np.abs(np.mean(error)) < 0.01
    assert np.max(error) < 0.1  # nothing too far off
    assert np.mean(error) < 0.05  # mean is fairly close
    # for sparse matrix
    transformer.dense_output = True
    X_trans_sp = transformer.transform(csr_matrix(X))
    assert_allclose(X_trans, X_trans_sp)
