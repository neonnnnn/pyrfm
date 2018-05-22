import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from math import sqrt


class RandomFourier(BaseEstimator, TransformerMixin):
    def __init__(self, D=100, kernel='rbf', gamma='auto', mode='cos',
                 random_state=None):
        self.D = D
        self.gamma = gamma
        self.kernel = kernel
        self.mode = mode
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        d = check_array(X, True).shape[1]

        if self.mode == 'cos':
            D = self.D
        else:
            D = self.D / 2

        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        # ToDo: Implement other shift-invariant kernels
        if self.kernel in ['rbf', 'gaussian']:
            self.Omega_ = random_state.normal(scale=np.sqrt(gamma*2),
                                              size=(d, D))
        else:
            raise ValueError('Kernel {} is not supported.'
                             'Use "rbf" or "Gaussian"'.format(self.kernel))
        if self.mode == 'cos':
            self.b_ = random_state.uniform(0, 2*np.pi, size=self.D)
        else:
            self.b_ = 0.

        return self

    def transform(self, raw_X):
        raw_X = check_array(raw_X, True)
        output = safe_sparse_dot(raw_X, self.Omega_, True) + self.b_
        if self.mode == 'cos':
            output = np.cos(output)
        else:
            output = np.hstack((np.cos(output), np.sin(output)))
        return sqrt(2./self.D) * output
