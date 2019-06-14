import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import RandomFourier
from sklearn.metrics.pairwise import rbf_kernel
# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)


def test_random_fourier():
    # compute exact kernel
    for gamma in [0.1, 1, 10]:
        kernel = rbf_kernel(X, Y, gamma)
        # approximate kernel mapping
        rf_transform = RandomFourier(n_components=6000, gamma=gamma,
                                     use_offset=True, random_state=0)
        X_trans = rf_transform.fit_transform(X)
        Y_trans = rf_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.01)
        assert_less_equal(np.max(error), 0.1)  # nothing too far off
        assert_less_equal(np.mean(error), 0.05)  # mean is fairly close

        rf_transform = RandomFourier(n_components=6000, gamma=gamma,
                                     use_offset=False, random_state=0)
        X_trans = rf_transform.fit_transform(X)
        Y_trans = rf_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.01)
        assert_less_equal(np.max(error), 0.1)  # nothing too far off
        assert_less_equal(np.mean(error), 0.05)  # mean is fairly close
