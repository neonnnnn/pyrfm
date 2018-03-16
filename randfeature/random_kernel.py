import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from math import sqrt
from .kernel_fast import anova, all_subset


def _anova(degree=2):
    def __anova(X, P):
        return anova(X, P, degree)

    return __anova


class RandomKernelProduct(BaseEstimator, TransformerMixin):
    def __init__(self, D=100, kernel='anova', degree=2, random_state=None):
        self.D = D
        self.degree = degree
        self.random_state = random_state

        if isinstance(kernel, str):
            if kernel == 'anova':
                self.kernel = _anova(self.degree)
            elif kernel == 'all_subset':
                self.kernel = all_subset
        else:
            self.kernel = kernel

    def fit(self, X, y=None):
        random_state = check_random_state(self.andom_state)
        d = check_array(X, ['csr']).shape[1]
        self.Projs = random_state.randint(2, size=(self.D, d))*2-1
        return self

    def transform(self, raw_X):
        n, d = check_array(raw_X, ['csr']).shape
        output = self.kernel(raw_X, self.Projs).astype(np.float64)
        output /= sqrt(self.D)
        return output
