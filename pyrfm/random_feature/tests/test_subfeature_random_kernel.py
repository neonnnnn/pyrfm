import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import (assert_almost_equal,
                                   assert_allclose_dense_sparse)
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import anova, SubfeatureRandomKernel, RandomKernel
import pytest
import itertools


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)

X_sp = rng.random_sample(size=(300, 2000))
Y_sp = rng.random_sample(size=(300, 2000))
X_sp[X_sp<0.99] = 0
Y_sp[Y_sp<0.99] = 0
X_sp /= np.sum(X_sp, axis=1, keepdims=True)
Y_sp /= np.sum(Y_sp, axis=1, keepdims=True)

X_sp = csr_matrix(X_sp)
Y_sp = csr_matrix(Y_sp)

degrees = [2, 3, 4, 5]
kernels = ['anova', 'anova_cython']
params = itertools.product(degrees, kernels)


@pytest.mark.parametrize("degree, kernel", params)
def test_anova_kernel(degree, kernel):
    # compute exact kernel
    gram = anova(X, Y, degree)
    # approximate kernel mapping
    rk_transform = SubfeatureRandomKernel(n_components=1000,
                                          random_state=rng,
                                          kernel=kernel, degree=degree,
                                          distribution="rademacher",
                                          n_sub_features=25)
    X_trans = rk_transform.fit_transform(X)
    Y_trans = rk_transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)

    error = gram - kernel_approx
    assert np.abs(np.mean(error)) < 0.001
    assert np.max(error) < 0.01  # nothing too far off
    assert np.mean(error) < 0.005  # mean is fairly close


@pytest.mark.parametrize("degree", [2,3,4,5])
def test_anova_kernel_sparse_subset(degree):
    # compute exact kernel
    n_components = 2000*5
    n_sub_features = 25
    gram = anova(X_sp, Y_sp, degree, True)
    # approximate kernel mapping
    rk_transform = SubfeatureRandomKernel(n_components=n_components,
                                          random_state=rng,
                                          kernel='anova', degree=degree,
                                          distribution="rademacher",
                                          n_sub_features=n_sub_features)
    X_trans = rk_transform.fit_transform(X_sp)
    Y_trans = rk_transform.transform(Y_sp)
    assert_almost_equal(rk_transform.random_weights_.nnz,
                        n_components*n_sub_features)

    kernel_approx = safe_sparse_dot(X_trans, Y_trans.T, dense_output=True)
    error = gram - kernel_approx
    assert np.abs(np.mean(error)) < 0.001
    assert np.max(error) < 0.1  # nothing too far off
    assert np.mean(error) < 0.005  # mean is fairly close
    assert_almost_equal(n_sub_features*n_components,
                        rk_transform.random_weights_.nnz)
    assert_allclose_dense_sparse(
        n_sub_features*np.ones(n_components),
        np.array(abs(rk_transform.random_weights_).sum(axis=0))[0]
    )
