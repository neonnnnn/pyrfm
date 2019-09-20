import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from math import sqrt
from .utils import get_random_matrix, next_pow_of_two
from scipy.linalg import qr_multiply
from scipy.stats import chi
import warnings


def _get_random_matrix(distribution):
    return lambda rng, size: get_random_matrix(rng, distribution, size)


class OrthogonalRandomFeature(BaseEstimator, TransformerMixin):
    """Approximates feature map of the RBF or dot kernel by Monte Carlo
    approximation by Orthogonal Random Feature map.

    Parameters
    ----------
        n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.
        If n_components is not a n-tuple of n_features, it is automatically
        changed to the smallest n-tuple of the n_features that is bigger than
        n_features, which is bigger than n_components.
        That is, ceil(n_components/n_features)*n_features.

    gamma : float (default=0.5)
        Band width parameter. gamma = 1/2\sigma^2, where \sigma is a std
        parameter for gaussian distribution.

    distribution : str or function (default="gaussian")
        A function for sampling random basis whose arguments
        are random_state and size.
        Its arguments must be random_state and size.
        For str, "gaussian" (or "normal"), "rademacher", "laplace", or
        "uniform" can be used.

    random_fourier : boolean (default=True)
        Approximate RBF kernel or not.
        If True, Fastfood samples random_offset_ in the fit method and computes
        the cosine of structured_matrix-feature_vector product + random_offset_
        in transform.
        If False, Fastfood does not sample it and computes just
        structured_matrix-feature_vector product (i.e., approximates dot product
        kernel).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    random_weights_ : array, shape (n_components, n_features) (use_offset=True)
                                   or (n_components/2, n_features) (otherwise)
        The sampled basis.

    random_offset_ : array or None, shape (n_components, )
        The sampled offset vector. If use_offset=False, offset_=None.

    References
    ----------
    [1] Orthogonal Random Features
    Felix Xinnan Yu Ananda Theertha Suresh Krzysztof Choromanski
    Daniel Holtmann-Rice Sanjiv Kumar
    In NIPS 2016.
    (https://arxiv.org/pdf/1610.09072.pdf)
    """
    def __init__(self, n_components=100,  gamma=0.5, distribution="gaussian",
                 random_fourier=True, random_state=None):
        self.n_components = n_components
        self.distribution = distribution
        self.gamma = gamma
        self.random_fourier = random_fourier
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        n_stacks = int(np.ceil(self.n_components/n_features))
        n_components = n_stacks * n_features
        if n_components != self.n_components:
            warnings.warn("n_components is changed from {0} to {1}."
                          "You should set n_components n-tuple of the next"
                          "power of two of n_features."
                          .format(self.n_components, n_components))
            self.n_components = n_components

        size = (n_features, n_features)
        if isinstance(self.distribution, str):
            self.distribution = _get_random_matrix(self.distribution)
        random_weights_ = []
        for _ in range(n_stacks):
            W = self.distribution(random_state, size)
            S = np.diag(chi.rvs(df=n_features, size=n_features))
            WS, _ = qr_multiply(W, S)
            random_weights_ += [WS]
        self.random_weights_ = np.vstack(random_weights_)

        if self.random_fourier:
            self.random_offset_ = random_state.uniform(0, 2*np.pi,
                                                       size=self.n_components)
        else:
            self.random_offset_ = None

        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        output = safe_sparse_dot(X, self.random_weights_.T, True)
        if self.random_fourier:
            if self.gamma == 'auto':
                gamma = 1.0 / X.shape[1]
            else:
                gamma = self.gamma
            output = np.cos(np.sqrt(2*gamma)*output+self.random_offset_)
            output *= np.sqrt(2)
        return output / sqrt(self.n_components)


class StructuredOrthogonalRandomFeature(BaseEstimator, TransformerMixin):
    """Approximates feature map of the RBF or dot kernel by Monte Carlo
    approximation by Structured Orthogonal Random Fourier Feature map.

    Parameters
    ----------
        n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.
        If n_components is not a n-tuple of n_features, it is automatically
        changed to the smallest n-tuple of the n_features that is bigger than
        n_features, which is bigger than n_components.
        That is, ceil(n_components/n_features)*n_features.

    gamma : float (default=0.5)
        Band width parameter. gamma = 1/2\sigma^2, where \sigma is a std
        parameter for gaussian distribution.

    distribution : str or function (default="rademacher")
        A function for sampling random basis whose arguments
        are random_state and size.
        Its arguments must be random_state and size.
        For str, "gaussian" (or "normal"), "rademacher", "laplace", or
        "uniform" can be used.

    random_fourier : boolean (default=True)
        Approximate RBF kernel or not.
        If True, Fastfood samples random_offset_ in the fit method and computes
        the cosine of structured_matrix-feature_vector product + random_offset_
        in transform.
        If False, Fastfood does not sample it and computes just
        structured_matrix-feature_vector product (i.e., approximates dot product
        kernel).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    random_weights_ : array, shape (n_components, n_features) (use_offset=True)
                                   or (n_components/2, n_features) (otherwise)
        The sampled basis.

    random_offset_ : array or None, shape (n_components, )
        The sampled offset vector. If use_offset=False, offset_=None.

    References
    ----------
    [1] Orthogonal Random Features
    Felix Xinnan Yu Ananda Theertha Suresh Krzysztof Choromanski
    Daniel Holtmann-Rice Sanjiv Kumar
    In NIPS 2016.
    (https://arxiv.org/pdf/1610.09072.pdf)
    """
    def __init__(self, n_components=100,  gamma=0.5, distribution="rademacher",
                 random_fourier=True, random_state=None):
        self.n_components = n_components
        self.distribution = distribution
        self.gamma = gamma
        self.random_fourier = random_fourier
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        n_features_padded = next_pow_of_two(n_features)
        n_stacks = int(np.ceil(self.n_components/n_features_padded))
        n_components = n_stacks * n_features_padded

        if n_components != self.n_components:
            warnings.warn("n_components is changed from {0} to {1}."
                          "You should set n_components n-tuple of the next"
                          "power of two of n_features."
                          .format(self.n_components, n_components))
            self.n_components = n_components

        # n_stacks * n_features_padded = self.n_components
        if isinstance(self.distribution, str):
            self.distribution = _get_random_matrix(self.distribution)
        size = (n_stacks, n_features+2*n_features_padded)
        self.random_weights_ = self.distribution(random_state, size)
        if self.random_fourier:
            self.random_offset_ = random_state.uniform(0, 2*np.pi,
                                                       size=self.n_components)
        else:
            self.random_offset_ = None
        return self

    def transform(self, X):
        from .random_features_fast import transform_all_fast
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        Z = transform_all_fast(X, self)
        return Z