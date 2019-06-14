import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import TensorSketch


def polynomial(X, Y, degree):
    return safe_sparse_dot(X, Y.T, True)**degree

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)


def test_tensor_sketching():
    # compute exact kernel
    for degree in range(2, 5):
        kernel = polynomial(X, Y, degree)
        # approximate kernel mapping
        ts_transform = TensorSketch(n_components=1000, degree=degree,
                                    random_state=rng)
        X_trans = ts_transform.fit_transform(X)
        Y_trans = ts_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.001)
        assert_less_equal(np.max(error), 0.01)  # nothing too far off
        assert_less_equal(np.mean(error), 0.005)  # mean is fairly close