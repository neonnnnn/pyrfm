# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from math import sqrt
from .utils import get_random_matrix


def _get_random_matrix(distribution):
    return lambda rng, size, p: get_random_matrix(rng, distribution, size, p)


class RandomProjection(BaseEstimator, TransformerMixin):
    """Approximates feature map of the dot product kernel by Monte Carlo
    approximation by Random Projection`.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    distribution : str or function (default="rademacher")
        A function for sampling random basis whose arguments
        are random_state and size.
        Its arguments must be random_state and size.
        If None, the Rademacher distribution is used.

    p_sparse : float (default="auto")
        Sparsity parameter for "sparse_rademacher" distribution.
        If p_sparse = 0, "sparse_rademacher" is equivalent to "rademacher".
        The relationship between p_sparse and s in [1] is s = 1/(1-p).
        If auto, p_sparse = 1 - 1/sqrt(n_features), recommended in [1].

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_weights_ : array, or sparse matrix, shape (n_features, n_components)
        The sampled basis.

    References
    -----------
    [1] Very Sparse Random Projections.
    Ping Li, T. Hastie, and K. W. Church.
    In KDD 2006.
    (https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf)
    """

    def __init__(self, n_components=100, distribution="rademacher",
                 p_sparse="auto", random_state=None):
        self.n_components = n_components
        self.distribution = distribution
        self.p_sparse = p_sparse
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
        size = (n_features, self.n_components)
        if self.p_sparse == "auto":
            p_sparse = 1 - 1./np.sqrt(n_features)
        else:
            if isinstance(self.p_sparse, float):
                if 1. > self.p_sparse >= 0:
                    p_sparse = self.p_sparse
                else:
                    raise ValueError("p_sparse must be in [0, 1), but got {}"
                                     .format(self.p_sparse))
            else:
                raise TypeError("p_sparse is 'auto' or float in [0, 1), but "
                                "got type {}".format(type(self.p_sparse)))

        if isinstance(self.distribution, str):
            self.distribution = _get_random_matrix(self.distribution)
        self.random_weights_ = self.distribution(random_state, size, p_sparse)
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
        output = safe_sparse_dot(X, self.random_weights_, True)
        return output / sqrt(self.n_components)
