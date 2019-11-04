# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from .utils_fast import fisher_yates_shuffle
from .utils import next_pow_of_two, rademacher, get_random_matrix
from scipy.stats import chi
import warnings


def _get_random_matrix(distribution):
    return lambda rng, size: get_random_matrix(rng, distribution, size)


class FastFood(BaseEstimator, TransformerMixin):
    """Approximates feature maps of the product between random matrix and
    feature vectors by FastFood.

    This class can be used not only for approximating RBF kernel but
    also as a sub-routine for approximating the product between random matrix
    and feature vectors in some random features.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.
        If n_components is not a n-tuple of power of two, it is automatically
        changed to the smallest n-tuple of the smallest power of two number
        that is bigger than n_features, which is bigger than n_components.
        That is, ceil(n_components/2**{p})*2**{p}, where
        p = ceil(\log_2 (n_features)).

    gamma : float (default=0.5)
        Bandwidth parameter. gamma = 1/2\sigma^2, where \sigma is a std
        parameter for the Gaussian distribution.

    distribution : str or function (default="gaussian")
        A function for sampling random basis whose arguments
        are random_state and size.
        Its arguments must be random_state and size.
        For str, "gaussian" (or "normal"), "rademacher", "laplace", or
        "uniform" can be used.

    random_fourier : boolean (default=True)
        Whether to approximate the RBF kernel or not.
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
    random_weights_ : array, shape (T, n_feature_padded)
        The sampled basis, where T = np.ceil(n_components/n_features) and
        n_feature_padded = 2**np.ceil(np.log2(n_features))

    random_perm_: array, shape (T, n_features_padded)
        The sampled permutation matrix.

    random_sign_: array, shape (T, n_features)
        The sampled siged matrix.

    random_scaling_ : array, shape (T, n_features_padded)
        The sampled scaling matrix.

    random_offset_ : array, shape (n_components)
        The sampled random offset for random fourier features.
        If self.random_fouier is False, random_offset_ is None.

    References
    ----------
    [1] Fastfood — Approximating Kernel Expansions in Loglinear Time.
    Quoc Le, Tam´as Sarl´os, and Alex Smola.
    In ICML 2013.
    (http://proceedings.mlr.press/v28/le13-supp.pdf)

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
        n_features_padded = next_pow_of_two(n_features)
        n_stacks = int(np.ceil(self.n_components/n_features_padded))
        n_components = n_stacks * n_features_padded

        if n_components != self.n_components:
            warnings.warn("n_components is changed from {0} to {1}. "
                          "You should set n_components n-tuple of the next "
                          "power of two of n_features."
                          .format(self.n_components, n_components))
            self.n_components = n_components

        # n_stacks * n_features_padded = self.n_components
        size = (n_stacks, n_features_padded)
        if isinstance(self.distribution, str):
            self.distribution = _get_random_matrix(self.distribution)
        self.random_weights_ = self.distribution(random_state, size)
        self.random_sign_ = rademacher(random_state, (n_stacks, n_features))

        self.random_perm_ = np.zeros(size, dtype=np.int32)
        self._fy_vector_ = np.zeros(size, dtype=np.int32)
        for t in range(n_stacks):
            perm, fyvec = fisher_yates_shuffle(n_features_padded, random_state)
            self.random_perm_[t] = perm
            self._fy_vector_[t] = fyvec

        Frobs = np.sqrt(np.sum(self.random_weights_**2, axis=1, keepdims=True))
        self.random_scaling_ = chi.rvs(n_features_padded, size=size,
                                       random_state=random_state) / Frobs

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
        from .random_features_fast import transform_all_fast
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        Z = transform_all_fast(X, self)
        return Z
