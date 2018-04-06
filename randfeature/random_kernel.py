import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from math import sqrt
from .kernels import anova, all_subset


def _anova(degree=2):
    def __anova(X, P):
        return anova(X, P, degree)

    return __anova


def dot():
    def _dot(X, Y):
        return safe_sparse_dot(X, Y.T, True)

    return _dot


def get_random_matrix(rng, distribution, size):
    if distribution == 'rademacher':
        return rng.randint(2, size)*2 - 1
    elif distribution in ['gaussian', 'normal']:
        return rng.normal(0, 1, size)
    elif distribution == 'uniform':
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    elif distribution == 'laplace':
        return rng.laplace(0, 1./np.sqrt(2), size)
    else:
        raise ValueError('{} distribution is not implemented. Please use'
                         'rademacher, gaussian (normal), uniform or laplace.'
                         .format(distribution))


class RandomKernel(BaseEstimator, TransformerMixin):
    def __init__(self, D=100, kernel='anova', degree=2,
                 distribution='rademacher', random_state=None):
        self.D = D
        self.degree = degree
        self.random_state = random_state
        self.distribution = distribuion
        self.kernel = kernel

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        d = check_array(X, ['csr']).shape[1]
        if isinstance(self.kernel, str):
            if self.kernel == 'anova':
                self._kernel = _anova(self.degree)
            elif self.kernel == 'all_subset':
                self._kernel = all_subset
            elif self.kernel in ['dot', 'poly']:
                self._kerel = dot()
        else:
            self._kernel = kernel
        size = (self.D, d)
        distribution = self.distribution.lower()
        if self.kernel == 'poly':
            self.Projs = [get_random_matrix(random_state, distribution, size)
                          for _ in range(self.degree)]
        else:
            self.Projs = get_random_matrix(random_state, distribution, size)

        return self

    def transform(self, raw_X):
        n, d = check_array(raw_X, ['csr']).shape
        if self.kernel == 'poly':
            output = self._kernel(raw_X, self.Pros[0]).astype(np.float64)
            for proj in self.Projs[1:]:
                output *= self._kernel(raw_X, proj).astype(np.float64)
        else:
            output = self._kernel(raw_X, self.Projs).astype(np.float64)

        output /= sqrt(self.D)
        return output
