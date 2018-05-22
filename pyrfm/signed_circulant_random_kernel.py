import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from scipy.fftpack import fft, ifft
from math import sqrt
from .kernels import safe_power


class SignedCirculantRandomKernel(BaseEstimator, TransformerMixin):
    def __init__(self, t=10, kernel='anova', degree=2, random_state=None):
        self.t = t
        self.degree = degree
        self.random_state = random_state
        self.kernel = kernel

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        d = check_array(X, ['csr']).shape[1]
        self.projs_ = fft(random_state.randint(2, size=(self.t, d))*2-1)
        self.signs_ = random_state.randint(2, size=(self.t, d))*2-1
        return self

    def transform(self, raw_X):
        n, d = check_array(raw_X, ['csr']).shape
        output = []
        fft_X = fft(raw_X)
        D2_ = np.sum(safe_power(raw_X, 2), axis=1, keepdims=True)

        if self.degree == 3:
            fft_X3 = fft(safe_power(raw_X, 3))

        for proj, sign in zip(self.projs_, self.signs_):
            proj_x = ifft(fft_X*proj).real*sign
            output.append(safe_power(proj_x, self.degree))
            if self.degree == 3:
                output[-1] -= 3*proj_x*D2_
                output[-1] += 2*ifft(fft_X3*proj).real*sign

        output = np.asarray(output).real
        output = output.swapaxes(0,1).reshape(n, self.t*d)
        if self.degree == 2:
            output -= D2_
            output /= 2
        elif self.degree == 3:
            output /= 6

        return output/sqrt(self.t*d)
