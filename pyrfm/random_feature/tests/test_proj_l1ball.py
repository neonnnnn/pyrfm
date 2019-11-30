import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from pyrfm.random_feature import proj_l1ball, proj_l1ball_sort

# generate data
rng = np.random.RandomState(0)
n_samples = 400
n_features = 10
X = rng.rand(n_samples, n_features)
Y = rng.rand(n_samples, n_features)

def test_proj_l1ball():
    for x in X:
        assert_almost_equal(proj_l1ball(x, 1), proj_l1ball_sort(x, 1))
        assert_almost_equal(np.sum(proj_l1ball(x, 2)), 2)

        assert_almost_equal(proj_l1ball(x, 5), proj_l1ball_sort(x, 5))
        assert_almost_equal(np.sum(proj_l1ball(x, 2)), 2)