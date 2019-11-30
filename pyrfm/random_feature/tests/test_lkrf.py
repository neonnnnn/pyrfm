import numpy as np

from sklearn.utils.testing import assert_greater_equal, assert_almost_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import RBFSampler
from pyrfm import kernel_alignment, LearningKernelwithRandomFeature
from pyrfm import (RandomKernel, RandomFourier, RandomMaclaurin,
                   OrthogonalRandomFeature)
import pytest
data = load_digits(2, True)
X = data[0]
X /= np.max(X)
y = data[1]
y = 2 * y - 1


def _test_learning_kernel_with_random_feature(divergence, trans=None, rho=1):
    if trans is None:
        trans = RBFSampler(gamma=1,  random_state=0)
    trans.set_params(n_components=128)
    X_trans = trans.fit_transform(X)
    score = kernel_alignment(np.dot(X_trans, X_trans.T), y, False)
    lkrf = LearningKernelwithRandomFeature(trans, warm_start=False,
                                           divergence=divergence,
                                           eps_abs=1e-6, eps_rel=1e-6,
                                           max_iter=100, rho=rho)
    X_trans = lkrf.fit_transform(X, y)
    score_lkrf = kernel_alignment(np.dot(X_trans, X_trans.T), y, False)
    assert_greater_equal(score_lkrf, score)
    assert_almost_equal(np.sum(lkrf.importance_weights_), 1)
    assert_greater_equal(np.min(lkrf.importance_weights_), 0)
 
    # weak constrain: rho = 10*rho
    trans.fit(X)
    lkrf = LearningKernelwithRandomFeature(trans, warm_start=False,
                                           divergence=divergence, 
                                           eps_abs=1e-6, eps_rel=1e-6,
                                           max_iter=100, rho=rho*20)
    X_trans = lkrf.fit_transform(X, y)
    score_lkrf_weak = kernel_alignment(np.dot(X_trans, X_trans.T), y, False)
    print(score_lkrf_weak, score_lkrf, score)
    assert_greater_equal(score_lkrf_weak, score_lkrf)

    # remove bases
    n_nz = np.sum(lkrf.importance_weights_ != 0)
    print(n_nz)

    if lkrf.remove_bases():
        X_trans_removed = lkrf.transform(X)
        assert_almost_equal(X_trans_removed.shape[1], n_nz)
        indices = np.nonzero(lkrf.importance_weights_)[0]
        assert_almost_equal(X_trans_removed, X_trans[:, indices])
 

params =  [
    RBFSampler(n_components=128, random_state=0),   
    RandomFourier(n_components=128, random_state=0),   
    RandomFourier(n_components=128, random_state=0, use_offset=True),
    OrthogonalRandomFeature(n_components=128, random_state=0), 
    OrthogonalRandomFeature(n_components=128, random_state=0,
                            use_offset=True),
    RandomMaclaurin(random_state=0),
    RandomKernel(random_state=0)
]


@pytest.mark.parametrize("transformer", params)
def test_lkrf_chi2(transformer, rho=1):
    _test_learning_kernel_with_random_feature('chi2', transformer, rho)


def test_lkrf_chi2_origin():
    _test_learning_kernel_with_random_feature('chi2_origin')


@pytest.mark.parametrize("transformer", params)
def test_lkrf_kl(transformer):
    _test_learning_kernel_with_random_feature('kl', transformer, rho=0.01)


@pytest.mark.parametrize("transformer", params)
def test_lkrf_reverse_kl(transformer):
    _test_learning_kernel_with_random_feature('reverse_kl', transformer, 0.01)


@pytest.mark.parametrize("transformer", params)
def test_lkrf_tv(transformer):
    _test_learning_kernel_with_random_feature('tv', transformer, rho=.2)


@pytest.mark.parametrize("transformer", params)
def test_lkrf_squared(transformer):
    _test_learning_kernel_with_random_feature('squared', transformer, 
                                               rho=0.01)