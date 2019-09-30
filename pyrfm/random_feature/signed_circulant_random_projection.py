# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import issparse
from .utils import rademacher, get_random_matrix
from scipy.fftpack import fft, ifft
import warnings


def _get_random_matrix(distribution):
    return lambda rng, size: get_random_matrix(rng, distribution, size)


class SignedCirculantRandomMatrix(BaseEstimator, TransformerMixin):
    """Approximates feature maps of the product between random matrix and
    feature vectors by signed circulant random matrix.
    This class can be used not only for approximating RBF kernel but
    also as a sub-routine for approximating the product between random matrix
    and feature vectors in some random features.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.
        If n_components is not a n-tuple of the n_features, it is automatically
        changed to the smallest n-tuple of the n_features.

    gamma : float (default=0.5)
        Band width parameter. gamma = 1/2\sigma^2, where \sigma is a std
        parameter for gaussian distribution.

    random_fourier : boolean (default=True)
        Approximate RBF kernel or not.
        If True, this class samples random_offset_ in the fit method and
        computes the cosine of structured_matrix-feature_vector product
        + random_offset_in transform.
        If False, this class does not sample it and computes just
        structured_matrix-feature_vector product (i.e., approximates dot product
        kernel).

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    distribution : str or function (default="gaussian")
        A function for sampling random basis whose arguments
        are random_state and size.
        Its arguments must be random_state and size.
        For str, "gaussian" (or "normal"), "rademacher", "laplace", or
        "uniform" can be used.

    Attributes
    ----------
    random_weights_ : array, shape (n_stacks, n_features)
        The sampled basis, where n_stacks = np.ceil(n_components/n_features) and
        n_feature_padded = 2**np.ceil(np.log2(n_features)).
        In fit function, random_weights are fast fourier transformed.

    random_sign_ : array, shape (n_stacks, n_features)
        The sampled signed matrix.

    random_offset_ : array, shape (n_components)
        The sampled random offset for random fourier features.
        If self.random_fouier is False, random_offset_ is None.

    References
    ----------
    [1] Random Feature Mapping with Signed Circulant Matrix Projection
    Chang Feng, Qinghua Hu, and Shizhong Liao.
    In IJCAI 2015.
    (https://www.ijcai.org/Proceedings/15/Papers/491.pdf)
    """

    def __init__(self, n_components=100,  gamma=0.5, distribution="gaussian",
                 random_fourier=True, random_state=None):
        self.n_components = n_components
        self.distribution = distribution
        self.gamma = gamma
        self.random_fourier = random_fourier
        self.random_state = random_state

    def fit(self, X, y=None):
        """Generate random weights according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        if isinstance(self.distribution, str):
            self.distribution = _get_random_matrix(self.distribution)

        n_stacks = int(np.ceil(self.n_components/n_features))
        n_components = n_stacks * n_features
        if n_components != self.n_components:
            warnings.warn("n_components is changed from {0} to {1}. "
                          "You should set n_components n-tuple of the "
                          "n_features."
                          .format(self.n_components, n_components))
            self.n_components = n_components

        # n_stacks * n_features= self.n_components
        size = (n_stacks, n_features)
        self.random_weights_ = fft(self.distribution(random_state, size))
        self.random_sign_ = rademacher(random_state, size)

        if self.random_fourier:
            self.random_offset_ = random_state.uniform(0, 2*np.pi,
                                                       self.n_components)
        else:
            self.random_offset_ = None
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        n_stacks = self.random_weights_.shape[0]
        Z = np.zeros((n_samples, self.n_components))
        if issparse(X):
            from .random_features_fast import transform_all_fast
            Z = transform_all_fast(X, self)
            return Z
        else:
            fft_X = fft(X)
            for t, (fft_rw, sign) in enumerate(zip(self.random_weights_,
                                                   self.random_sign_)):
                projection = sign * ifft(fft_X * fft_rw).real
                Z[:, t * n_features:(t + 1) * n_features] = projection

            if self.random_fourier:
                Z = np.cos(Z*np.sqrt(2*self.gamma)+self.random_offset_)
                Z *= np.sqrt(2)
            return Z / np.sqrt(self.n_components)
