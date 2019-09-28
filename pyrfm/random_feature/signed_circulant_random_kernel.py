import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from scipy.fftpack import fft, ifft
from ..kernels import safe_power
import warnings
from math import sqrt
from scipy.sparse import issparse


class SignedCirculantRandomKernel(BaseEstimator, TransformerMixin):
    """Approximates feature map of the ANOVA kernel by Monte Carlo
    approximation by Signed Circulant Random Kernel map.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    degree : int (default=2)
        Parameter of the ANOVA kernel.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    [1] Random Feature Maps for the Itemset Kernel.
    Kyohei Atarashi, Subhransu Maji, and Satoshi Oyama
    In AAAI 2019.
    (https://www.aaai.org/ojs/index.php/AAAI/article/view/4188)
    """
    def __init__(self, n_components, kernel='anova', degree=2,
                 random_state=None):
        self.n_components = n_components
        self.degree = degree
        self.random_state = random_state
        self.kernel = kernel

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape

        if n_features > self.n_components:
            raise ValueError('n_components is lower than X.shape[1]')

        t = self.n_components//n_features
        size = (t, n_features)
        self.n_components_actual_ = t * n_features
        self.random_weights_ = fft(random_state.randint(2, size=size)*2-1)
        self.signs_ = random_state.randint(2, size=size)*2-1
        return self

    def transform(self, X):
        check_is_fitted(self, ["signs_", "random_weights_"])
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        output = []
        if issparse(X):
            fft_X = fft(X.toarray())
        else:
            fft_X = fft(X)
        fft_X_pow_3 = None
        if issparse(X):
            D2 = np.array(np.sum(safe_power(X, 2), axis=1))
        else:
            D2 = np.sum(safe_power(X, 2), axis=1, keepdims=True)

        t = self.n_components//n_features
        if self.n_components % n_features != 0:
            warnings.warn("self.n_components is indivisible by n_features. "
                          "Output.shape[1] is {}".format(t*n_features))
        if self.degree == 3:
            fft_X_pow_3 = fft(safe_power(X, 3, dense_output=True))

        for random_weight, sign in zip(self.random_weights_, self.signs_):
            random_weight_x = ifft(fft_X*random_weight).real*sign
            output.append(safe_power(random_weight_x, self.degree))
            if self.degree == 3:
                output[-1] -= 3*random_weight_x*D2
                output[-1] += 2*ifft(fft_X_pow_3*random_weight).real*sign

        output = np.asarray(output).real
        output = output.swapaxes(0, 1).reshape(n_samples, t*n_features)
        if self.degree == 2:
            output -= D2
            output /= 2
        elif self.degree == 3:
            output /= 6

        return output/sqrt(t*n_features)
