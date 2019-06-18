import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from math import sqrt


class RandomFourier(BaseEstimator, TransformerMixin):
    """Approximates feature map of the RBF kernel by Monte Carlo
    approximation by Random Fourier Feature map.

    Parameters
    ----------
    n_components : int
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.
    
    gamma : float or str
        Parameter for the RBF kernel.
        
    kernel : str
        Kernel to be approximated.
        "anova", "dot", or "all-subsets" can be used.

    use_offset : bool
        
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    [1] Random Features for Large-Scale Kernel Machines
    Ali Rahimi and Ben Recht
    In NIPS 2007
    (https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
    """
    def __init__(self, n_components=100, kernel='rbf', gamma='auto',
                 use_offset=False, random_state=None):
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.use_offset = use_offset
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape

        if self.use_offset:
            n_components = self.n_components
        else:
            n_components = int(self.n_components / 2)

        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        size = (n_features, n_components)
        # TODO: Implement other shift-invariant kernels
        if self.kernel in ['rbf', 'gaussian']:
            self.random_weights_ = random_state.normal(size=size,
                                                       scale=np.sqrt(2*gamma))
        else:
            raise ValueError('Kernel {} is not supported.'
                             'Use "rbf" or "Gaussian"'.format(self.kernel))
        if self.use_offset:
            self.offset_ = random_state.uniform(0, 2*np.pi,
                                                size=self.n_components)
        else:
            self.offset_ = 0.

        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        output = safe_sparse_dot(X, self.random_weights_, True)
        if self.use_offset:
            output += self.offset_
            output = np.cos(output)
        else:
            output = np.hstack((np.cos(output), np.sin(output)))
        return sqrt(2./self.n_components) * output
