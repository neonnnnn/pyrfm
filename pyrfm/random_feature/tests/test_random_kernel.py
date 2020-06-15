import numpy as np
from scipy.sparse import csr_matrix, issparse

from sklearn.utils.testing import (assert_allclose_dense_sparse,
                                   assert_almost_equal)
                                   
from pyrfm import anova, all_subsets, RandomKernel
import pytest
import itertools


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


distributions = ['rademacher', 'gaussian', 'laplace', 'uniform',
                 'sparse_rademacher']
degrees = [2, 3, 4, 5]
kernels = ['anova', 'anova_cython']
params = itertools.product(distributions, degrees, kernels)


@pytest.mark.parametrize("dist,degree,kernel", params)
def test_anova_kernel(dist, degree, kernel):
    # compute exact kernel
    gram = anova(X, Y, degree)
    # approximate kernel mapping
    rk_transform = RandomKernel(n_components=1000, random_state=0,
                                kernel=kernel, degree=degree,
                                distribution=dist, p_sparse=0.5)

    X_trans = rk_transform.fit_transform(X)
    Y_trans = rk_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    error =  gram - kernel_approx
    assert np.abs(np.mean(error)) < 0.0001
    assert np.max(error) < 0.001  # nothing too far off
    assert np.mean(error) < 0.0005  # mean is fairly close

    # sparse input
    X_trans_sp = rk_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)

    # sparse output
    if dist == "sparse_rademacher":
        rk_transform.dense_output = False
        X_trans_sp = rk_transform.transform(X_sp)
        assert issparse(X_trans_sp)
        assert_allclose_dense_sparse(X_trans, X_trans_sp.toarray())
    else:
        rk_transform.dense_output = False
        X_trans_sp = rk_transform.transform(X_sp)
        assert not issparse(X_trans_sp)
        assert_allclose_dense_sparse(X_trans, X_trans_sp)


@pytest.mark.parametrize("p_sparse",[0.9, 0.8, 0.7, 0.6, 0.5])
def test_sparse_rademacher(p_sparse):
    # approximate kernel mapping
    for p_sparse in [0.9, 0.8, 0.7, 0.6, 0.5]:
        rk_transform = RandomKernel(n_components=1000, random_state=0,
                                    kernel='anova',
                                    distribution='sparse_rademacher',
                                    p_sparse=p_sparse)
        X_trans = rk_transform.fit_transform(X)
        nnz_actual = rk_transform.random_weights_.nnz
        nnz_expected = X.shape[1]*rk_transform.n_components*(1-p_sparse)
        assert_almost_equal(np.abs(1-nnz_actual/nnz_expected), 0, 0.1)



@pytest.mark.parametrize("dist", distributions)
def test_all_subsets_kernel(dist):
    # compute exact kernel
    p_sparse = 0.5
    kernel = all_subsets(X, Y)
    # approximate kernel mapping
    rk_transform = RandomKernel(n_components=5000, random_state=0,
                                kernel='all_subsets',
                                distribution=dist, p_sparse=p_sparse)
    X_trans = rk_transform.fit_transform(X)
    Y_trans = rk_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = kernel - kernel_approx
    if dist == 'sparse_rademacher':
        nnz = rk_transform.random_weights_.nnz
        nnz_expect = np.prod(rk_transform.random_weights_.shape)*p_sparse
        nnz_var = np.sqrt(nnz_expect * (1-p_sparse))
        assert np.abs(nnz-nnz_expect) < 3*nnz_var
    assert np.abs(np.mean(error)) < 0.01
    assert np.max(error) < 0.1  # nothing too far off
    assert np.mean(error) < 0.05  # mean is fairly close

    X_trans_sp = rk_transform.transform(X_sp)
    assert_allclose_dense_sparse(X_trans, X_trans_sp)
