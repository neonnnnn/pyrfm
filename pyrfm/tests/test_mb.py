import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import MB, SparseMB
from pyrfm import intersection

# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))


def test_mb():
    # compute exact kernel
    kernel = intersection(X, Y)
    # approximate kernel mapping
    mb_transform = MB(n_components=10000)
    X_trans = mb_transform.fit_transform(X)
    Y_trans = mb_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    assert_less_equal(np.mean(np.abs(error)), 50 / (mb_transform.n_grids_))
