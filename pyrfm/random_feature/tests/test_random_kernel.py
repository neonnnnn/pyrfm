import numpy as np
from scipy.sparse import csr_matrix, issparse

from sklearn.utils.testing import (assert_less_equal,
                                   assert_allclose_dense_sparse,
                                   assert_almost_equal)
from pyrfm import anova, all_subsets, RandomKernel


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


def test_sparse_rademacher():
    # approximate kernel mapping
    for p_sparse in [0.9, 0.8, 0.7, 0.6, 0.5]:
        rk_transform = RandomKernel(n_components=1000, random_state=rng,
                                    kernel='anova',
                                    distribution='sparse_rademacher',
                                    p_sparse=p_sparse)
        X_trans = rk_transform.fit_transform(X)
        nnz_actual = rk_transform.random_weights_.nnz
        nnz_expected = X.shape[1]*rk_transform.n_components*(1-p_sparse)
        assert_almost_equal(np.abs(1-nnz_actual/nnz_expected), 0, 0.1)


def test_anova_kernel():
    # compute exact kernel
    distributions = ['rademacher', 'gaussian', 'laplace', 'uniform',
                     'sparse_rademacher']
    for degree in range(2, 5):
        kernel = anova(X, Y, degree)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomKernel(n_components=1000, random_state=rng,
                                        kernel='anova', degree=degree,
                                        distribution=dist, p_sparse=0.5)

            X_trans = rk_transform.fit_transform(X)
            Y_trans = rk_transform.transform(Y)
            kernel_approx = np.dot(X_trans, Y_trans.T)
            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.0001)
            assert_less_equal(np.max(error), 0.001)  # nothing too far off
            assert_less_equal(np.mean(error), 0.0005)  # mean is fairly close

            # sparse input
            X_trans_sp = rk_transform.transform(X_sp)
            assert_allclose_dense_sparse(X_trans, X_trans_sp)

            # sparse output
            rk_transform.dense_output = False
            X_trans_sp = rk_transform.transform(X_sp)
            if issparse(X_trans_sp):
                X_trans_sp = X_trans_sp.toarray()
            assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_all_subsets_kernel():
    # compute exact kernel
    distributions = ['rademacher', 'gaussian', 'laplace', 'uniform',
                     'sparse_rademacher']
    kernel = all_subsets(X, Y)
    for dist in distributions:
        # approximate kernel mapping
        rk_transform = RandomKernel(n_components=3000, random_state=rng,
                                    kernel='all_subsets',
                                    distribution=dist, p_sparse=0.5)
        X_trans = rk_transform.fit_transform(X)
        Y_trans = rk_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.01)
        assert_less_equal(np.max(error), 0.1)  # nothing too far off
        assert_less_equal(np.mean(error), 0.05)  # mean is fairly close

        X_trans_sp = rk_transform.transform(X_sp)
        assert_allclose_dense_sparse(X_trans, X_trans_sp)


def test_anova_cython_kernel():
    # compute exact kernel
    distributions = ['rademacher', 'gaussian', 'laplace', 'uniform',
                     'sparse_rademacher']
    for degree in range(2, 5):
        kernel = anova(X, Y, degree)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomKernel(n_components=1000, random_state=rng,
                                        kernel='anova_cython', degree=degree,
                                        distribution=dist, p_sparse=0.5)

            X_trans = rk_transform.fit_transform(X)
            Y_trans = rk_transform.transform(Y)
            kernel_approx = np.dot(X_trans, Y_trans.T)
            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.0001)
            assert_less_equal(np.max(error), 0.001)  # nothing too far off
            assert_less_equal(np.mean(error), 0.0005)  # mean is fairly close

            # sparse input
            X_trans_sp = rk_transform.transform(X_sp)
            assert_allclose_dense_sparse(X_trans, X_trans_sp)

            # sparse output
            rk_transform.dense_output = False
            X_trans_sp = rk_transform.transform(X_sp)
            assert_allclose_dense_sparse(X_trans, X_trans_sp.toarray())