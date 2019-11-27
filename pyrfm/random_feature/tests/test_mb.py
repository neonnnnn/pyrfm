import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from pyrfm import MB, SparseMB
from pyrfm import intersection
from sklearn.utils.extmath import safe_sparse_dot

# generate data
rng = np.random.RandomState(0)
n_samples = 300
n_features = 50
X = rng.random_sample(size=(n_samples, n_features))
Y = rng.random_sample(size=(n_samples, n_features))


def test_mb():
    for dense_output in [True, False]:
        # compute exact kernel
        kernel = intersection(X, Y)
        # approximate kernel mapping
        mb_transform = MB(n_components=10000, dense_output=dense_output)
        X_trans = mb_transform.fit_transform(X)
        Y_trans = mb_transform.transform(Y)
        kernel_approx = safe_sparse_dot(X_trans, Y_trans.T, dense_output)
        error = kernel - kernel_approx
        assert_less_equal(np.mean(np.abs(error)), 50 / mb_transform.n_grids_)

        # for sparse matrix
        X_trans_sp = mb_transform.fit_transform(csr_matrix(X))
        assert_allclose_dense_sparse(X_trans_sp, X_trans)
    
    mb_transform = MB(n_components=10000, dense_output=True)
    X_trans_dense = mb_transform.fit_transform(X)
    mb_transform = MB(n_components=10000, dense_output=False)
    X_trans_sparse = mb_transform.fit_transform(X)
    assert_allclose_dense_sparse(X_trans_sparse.toarray(), X_trans_dense)


def test_sparse_mb():
    mb_transform = SparseMB(n_components=10000)
    X_trans = mb_transform.fit_transform(X)
    assert_less_equal(X_trans.nnz, n_samples*n_features*2)
    assert_almost_equal(np.max(np.sum(X_trans, axis=1)), n_features, decimal=2)

    X_trans_sp = mb_transform.fit_transform(csr_matrix(X))
    assert_less_equal(X_trans.nnz, n_samples*n_features*2)
    assert_almost_equal(np.max(np.sum(X_trans, axis=1)), n_features, decimal=2)

    assert_almost_equal(X_trans.toarray(), X_trans_sp.toarray())
