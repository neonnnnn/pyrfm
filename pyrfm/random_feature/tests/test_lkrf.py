import numpy as np

from sklearn.utils.testing import assert_greater_equal, assert_almost_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import RBFSampler
from pyrfm import kernel_alignment, LearningKernelwithRandomFeature
from pyrfm import (RandomKernel, RandomFourier, RandomMaclaurin,
                   OrthogonalRandomFeature)

data = load_digits(2, True)
X = data[0]
X /= np.max(X)
y = data[1]
y = 2*y-1


def _test_learning_kernel_with_random_feature(divergence, trans=None, rho=1):
    print(divergence)
    if trans is None:
        trans = RBFSampler(gamma=1, n_components=500, random_state=0)
    X_trans = trans.fit_transform(X)
    score = kernel_alignment(np.dot(X_trans, X_trans.T), y)
    lkrf = LearningKernelwithRandomFeature(trans, warm_start=True,
                                           divergence=divergence,
                                           max_iter=100, rho=rho)
    X_trans = lkrf.fit_transform(X, y)
    score_lkrf = kernel_alignment(np.dot(X_trans, X_trans.T), y)
    print(score_lkrf, score)

    assert_greater_equal(score_lkrf, score)
    assert_almost_equal(np.sum(lkrf.importance_weights_), 1)
    assert_greater_equal(np.min(lkrf.importance_weights_), 0)
    # remove bases
    n_nz = np.sum(lkrf.importance_weights_ != 0)
    print(n_nz)
 
    if lkrf.remove_bases():
        X_trans_removed = lkrf.transform(X)
        assert_almost_equal(X_trans_removed.shape[1], n_nz)
        indices = np.nonzero(lkrf.importance_weights_)[0]
        assert_almost_equal(X_trans_removed, X_trans[:, indices])


def test_lkrf_chi2():
    _test_learning_kernel_with_random_feature('chi2', RBFSampler(n_components=128))
    _test_learning_kernel_with_random_feature('chi2', RBFSampler(n_components=128), rho=1)
    _test_learning_kernel_with_random_feature('chi2', RandomMaclaurin())
    _test_learning_kernel_with_random_feature('chi2', RandomMaclaurin(), rho=1)
    _test_learning_kernel_with_random_feature('chi2', RandomKernel())
    _test_learning_kernel_with_random_feature('chi2', RandomKernel(), rho=1)
    _test_learning_kernel_with_random_feature('chi2', RandomFourier(use_offset=True))
    _test_learning_kernel_with_random_feature('chi2', RandomFourier(use_offset=True), rho=1)
    _test_learning_kernel_with_random_feature('chi2',
                                               OrthogonalRandomFeature(use_offset=True))
    _test_learning_kernel_with_random_feature('chi2',
                                               OrthogonalRandomFeature(use_offset=True), rho=1)
    _test_learning_kernel_with_random_feature('chi2',
                                               OrthogonalRandomFeature(random_fourier=False))
    _test_learning_kernel_with_random_feature('chi2',
                                               OrthogonalRandomFeature(random_fourier=False), rho=1)

def test_lkrf_chi2_origin():
    _test_learning_kernel_with_random_feature('chi2_origin')


def test_lkrf_kl():
    _test_learning_kernel_with_random_feature('kl')


def test_lkrf_reverse_kl():
    _test_learning_kernel_with_random_feature('reverse_kl')


def test_lkrf_tv():
    _test_learning_kernel_with_random_feature('tv', rho=2)
    _test_learning_kernel_with_random_feature('tv', rho=1)


def test_lkrf_squared():
    _test_learning_kernel_with_random_feature('squared', rho=0.00001)
    _test_learning_kernel_with_random_feature('squared', rho=0.0001)