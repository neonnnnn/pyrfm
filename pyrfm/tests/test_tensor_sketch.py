import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import TensorSketch
from scipy.fftpack._fftpack import drfft


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


def test_tensor_sketching_cython():
    degree = 2
    ts = TensorSketch(n_components=1000, degree=degree,
                      random_state=rng)
    X_trans = ts.fit_transform(X)
    z = np.zeros(ts.n_components)
    z_cache = np.zeros(ts.n_components)
    X_trans_cython = np.zeros((X.shape[0], ts.n_components))
    hash_indices = ts.hash_indices_
    hash_signs = ts.hash_signs_
    n_features = X.shape[1]
    for i, x in enumerate(X):
        for j in range(ts.n_components):
            z[j] = 0
            z_cache[j] = 0

        for j in range(n_features):
            z[hash_indices[j]] += x[j]*hash_signs[j]

        drfft(z, direction=1, overwrite_x=True)

        for offset in range(n_features, n_features*degree, n_features):
            for j in range(n_features):
                z_cache[hash_indices[j+offset]] += x[j]*hash_signs[j+offset]

            drfft(z_cache, direction=1, overwrite_x=True)

            for j in range(ts.n_components):
                z[j] *= z_cache[j]
        drfft(z, direction=-1, overwrite_x=True)
        X_trans_cython[i] = np.array(z)
    assert_almost_equal(X_trans, X_trans_cython)
