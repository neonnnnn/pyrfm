import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.testing import assert_array_almost_equal, assert_allclose
from sklearn.utils.extmath import safe_sparse_dot

from itertools import product, combinations
from functools import reduce
import pyrfm
import pytest

rng = np.random.RandomState(0)
X = rng.random_sample(size=(10, 8))
Y = rng.random_sample(size=(10, 8))


def _product(x):
    return reduce(lambda a, b: a * b, x, 1)


def _power_iter(x, degree):
    return product(*([x] * degree))


def dumb_homogeneous(x, p, degree=2):
    return sum(_product(x[k] * p[k] for k in ix)
               for ix in _power_iter(range(len(x)), degree))


def dumb_anova(x, p, degree=2):
    return sum(_product(x[k] * p[k] for k in ix)
               for ix in combinations(range(len(x)), degree))


def dumb_all_subsets(x, p):
    return 1. + sum(_product(x[k] * p[k] for k in ix)
                    for degree in range(1, len(x)+1)
                    for ix in combinations(range(len(x)), degree))


def safe_power(X, degree, dense_output=False):
    if issparse(X):
        ret = X.power(degree)
        if dense_output and issparse(ret):
            return ret.toarray()
        else:
            return ret
    else:
        return X ** degree


def safe_np_elem_prod(X, Y, dense_output=False):
    if not (issparse(X) or issparse(Y)):
        return X*Y
    else:
        if issparse(X):
            ret = X.multiply(Y)
        else:
            ret = Y.multiply(X)

        if dense_output:
            return ret.toarray()
        else:
            return ret


def D(X, P, degree, dense_output=True):
    return safe_sparse_dot(safe_power(X, degree), safe_power(P, degree).T,
                           dense_output=dense_output)


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
def test_anova_kernel(degree):
    expected = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            expected[i, j] = dumb_anova(X[i], Y[j], degree=degree)

    anova = pyrfm.anova(X, Y, degree)
    assert_array_almost_equal(expected, anova, decimal=4)

    anova = pyrfm.anova_fast(X, Y, degree)
    assert_array_almost_equal(expected, anova, decimal=4)


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
def test_anova_kernel_sparse(degree):
    expected = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            expected[i, j] = dumb_anova(X[i], Y[j], degree=degree)

    anova = pyrfm.anova(csr_matrix(X), Y, degree, True)
    assert_array_almost_equal(expected, anova, decimal=4)

    anova = pyrfm.anova(csr_matrix(X), Y, degree, False)
    assert_array_almost_equal(expected, anova, decimal=4)
    
    anova = pyrfm.anova(X, csr_matrix(Y), degree, True)
    assert_array_almost_equal(expected, anova, decimal=4)

    anova = pyrfm.anova(X, csr_matrix(Y), degree, False)
    assert_array_almost_equal(expected, anova, decimal=4)

    anova = pyrfm.anova(csr_matrix(X), csr_matrix(Y), degree, True)
    assert_array_almost_equal(expected, anova, decimal=4)

    anova = pyrfm.anova(csr_matrix(X), csr_matrix(Y), degree, False)
    print(type(anova))
    assert_array_almost_equal(expected, anova.toarray(), decimal=4)


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
def test_anova_kernel_fast_sparse(degree):
    expected = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            expected[i, j] = dumb_anova(X[i], Y[j], degree=degree)

    anova = pyrfm.anova_fast(csr_matrix(X), Y, degree, False)
    assert_array_almost_equal(expected, anova.toarray(), decimal=4)

    anova = pyrfm.anova_fast(X, csr_matrix(Y), degree, False)
    assert_array_almost_equal(expected, anova.toarray(), decimal=4)

    anova = pyrfm.anova_fast(csr_matrix(X), csr_matrix(Y), degree, False)
    assert_array_almost_equal(expected, anova.toarray(), decimal=4)

def test_all_subsets_kernel():
    expected = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            expected[i, j] += dumb_all_subsets(X[i], Y[j])

    all_subsets = pyrfm.all_subsets(X, Y)
    assert_array_almost_equal(expected, all_subsets, decimal=4)


def test_score_l1():
    expected = 0
    # any kernels
    K = pyrfm.all_subsets(X, Y)
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            expected += abs(np.dot(X[i], X[j]) - K[i, j])
    expected /= (X.shape[0]*(X.shape[0]+1)/2)
    assert_allclose(pyrfm.score(X, K, 'l1'), expected)


def test_score_l2():
    expected = 0
    # any kernels
    K = pyrfm.all_subsets(X, Y)
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            print(expected)
            expected += (np.dot(X[i], X[j]) - K[i, j])**2
    expected /= (X.shape[0]*(X.shape[0]+1)/2)
    assert_allclose(pyrfm.score(X, K, 'l2'), expected)
