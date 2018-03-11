import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix
from scipy.special import factorial, binom


class RandomMaclaurin(BaseEstimator, TransformerMixin):
    def __init__(self, D=100, p=10, kernel='poly', degree=2, gamma='auto', bias=0.,
                 coefs=None, random_state=1, max_n=50, h01=False):
        self.D = D
        self.p = p
        self.gamma = gamma
        self.degree = degree
        self.coefs = coefs
        self.bias = float(bias)
        self.kernel = kernel
        self.random_state = random_state
        self.max_n = max_n
        self.p_choice = None
        self.h01 = h01

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        d = check_array(X, True).shape[1]

        if self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        if self.coefs is None:
            if self.kernel == 'poly':
                self.coefs = self.bias ** np.arange(self.degree+1)[::-1]
                self.coefs *= binom(self.degree,
                                    range(self.degree+1)).astype(np.float64)
                self.coefs /= factorial(self.degree+1)
                coefs = self.coefs
            elif self.kernel == 'exp':
                coefs = self._gamma ** np.arange(self.max_n)
                coefs /= factorial(range(self.max_n))
        else:
            coefs = self.coefs

        if self.h01:
            coefs[0, 1] = 0.

        if self.p_choice is None:
            p_choice = (1/self.p) ** (np.arange(len(self.coefs)) + 1)
            if np.sum(coefs==0.) != 0:
                p_choice[coefs==0] = 0
                p_choice /= np.sum(p_choice)
            self.p_choice = p_choice

        self.orders = random_state.choice(len(self.p_choice),
                                          self.D, p=self.p_choice)
        self.projs = [random_state.randint(2, size=(d, order))*2-1
                      for order in self.orders]

        return self

    def transform(self, raw_X):
        n, d = check_array(raw_X, True).shape
        output = np.empty((n, 0))
        if self.h01:
            output = np.hstack([raw_X, output])
            output = np.hstack((np.ones(n, 1), output))

        for proj in self.projs:
            if proj.shape[1] == 0:
                P = np.ones((n, 1))
            else:
                P = np.prod(safe_sparse_dot(raw_X, proj, True),
                            axis=1, keepdims=True)
            output = np.hstack((output, P))

        output *= np.sqrt(self.coefs[self.orders]/self.D)
        output /= np.sqrt(self.p_choice[self.orders])

        return output
