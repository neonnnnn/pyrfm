import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from math import sqrt


class RandomFourier(BaseEstimator, TransformerMixin):
    def __init__(self, D, kernel='rbf', gamma='auto', random_state=None):
        self.D = D
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = check_random_state(random_state)

    def fit(self, X, y=None):
        d = check_array(X, True).shape[1]

        if self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        if self.kernel in ['rbf', 'gaussian']:
            self.Omega = rng.normal(scale=np.sqrt(self._gamma*2), size=(d, self.D))
        else:
            raise ValueError('Kernel {} is not supported.'.format(self.kernel))
        self.b = rng.uniform(0, 2*np.pi, size=self.D)

        return self

    def transform(self, raw_X):
        n, d = check_array(raw_X, True).shape
        output = safe_sparse_dot(raw_X, self.Omega, True) + self.b
        output = np.cos(output) * sqrt(2./self.D)
        return output
