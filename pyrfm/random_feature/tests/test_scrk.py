import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.testing import (assert_less_equal,
                                   assert_allclose_dense_sparse)
from pyrfm import anova, SignedCirculantRandomKernel


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


def test_anova_kernel():
    # compute exact kernel
    for degree in range(2, 5):
        kernel = anova(X, Y, degree)
        # approximate kernel mapping
        rk_transform = SignedCirculantRandomKernel(n_components=1000,
                                                   random_state=rng,
                                                   kernel='anova',
                                                   degree=degree)
        X_trans = rk_transform.fit_transform(X)
        Y_trans = rk_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.0001)
        assert_less_equal(np.max(error), 0.001)  # nothing too far off
        assert_less_equal(np.mean(error), 0.0005)  # mean is fairly close

        X_trans_sp = rk_transform.transform(X_sp)
        assert_allclose_dense_sparse(X_trans, X_trans_sp)
