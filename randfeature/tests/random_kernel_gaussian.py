import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from math import sqrt
from randfeature.kernels import anova, all_subset


def _anova(degree=2):
    def __anova(X, P):
        return anova(X, P, degree)

    return __anova


def dot():
    def _dot(X, Y):
        return safe_sparse_dot(X, Y.T, True)

    return _dot


class RandomKernelGaussian(BaseEstimator, TransformerMixin):
    def __init__(self, D=100, kernel='anova', degree=2, random_state=None):
        self.D = D
        self.degree = degree
        self.random_state = random_state
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
                self._kerel = dot
        else:
            self._kernel = kernel

        if self.kernel == 'poly':
            self.Projs = [random_state.normal(0, 0.1, size=(self.D, d))
                          for _ in range(self.degree)]
        else:
            self.Projs = random_state.normal(0, 0.1, size=(self.D, d))

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
