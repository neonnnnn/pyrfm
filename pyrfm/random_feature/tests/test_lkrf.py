import numpy as np

from sklearn.utils.testing import assert_greater_equal, assert_almost_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import RBFSampler
from pyrfm import kernel_alignment, LearningKernelwithRandomFeature

data = load_digits(2, True)
X = data[0]
y = data[1]
y = 2*y-1


def _test_learning_kernel_with_random_feature(divergence, rho=1):
    print(divergence)
    trans = RBFSampler(gamma=1, n_components=500, random_state=0)
    X_trans = trans.fit_transform(X)
    score = kernel_alignment(np.dot(X_trans, X_trans.T), y)
    lkrf = LearningKernelwithRandomFeature(trans, warm_start=True,
                                            divergence=divergence,
                                            max_iter=100, rho=rho)
    X_trans = lkrf.fit_transform(X, y)
    score_lkrf = np.sum(np.dot(y, X_trans)**2)
    print(score_lkrf, score)
    print(np.sum(lkrf.importance_weights_ != 0))
    #print(lkrf.importance_weights_)
    assert_greater_equal(score_lkrf, score)
    assert_almost_equal(np.sum(lkrf.importance_weights_), 1)
    assert_greater_equal(np.min(lkrf.importance_weights_), 0)


def test_lkrf_chi2():
    _test_learning_kernel_with_random_feature('chi2')
    _test_learning_kernel_with_random_feature('chi2', rho=10)


def test_lkrf_chi2_origin():
    _test_learning_kernel_with_random_feature('chi2_origin')
    _test_learning_kernel_with_random_feature('chi2', rho=10)


def test_lkrf_kl():
    _test_learning_kernel_with_random_feature('kl')
    _test_learning_kernel_with_random_feature('kl', 10)

def test_lkrf_reverse_kl():
    _test_learning_kernel_with_random_feature('reverse_kl')
    _test_learning_kernel_with_random_feature('reverse_kl', 10)


def test_lkrf_tv():
    _test_learning_kernel_with_random_feature('tv', 0.001)
    _test_learning_kernel_with_random_feature('tv', 0.01)


def test_lkrf_squared():
    _test_learning_kernel_with_random_feature('squared', 0.001)
    _test_learning_kernel_with_random_feature('squared', 0.0001)
