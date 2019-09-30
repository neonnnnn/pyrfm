# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix, issparse
from math import sqrt
from ..kernels import anova, all_subsets, anova_fast, pairwise
from .utils import get_random_matrix
from .sparse_rademacher import get_subfeatures_indices
import warnings
from scipy.special import comb


def _anova(degree=2):
    return lambda X, Y, dense_output: anova(X, Y, degree, dense_output)


def _anova_fast(degree=2):
    return lambda X, Y, dense_output: anova_fast(X, Y, degree, dense_output)


def dot():
    return lambda X, Y, dense_output: safe_sparse_dot(X, Y.T, dense_output)


def _pairwise(symmetric=False):
    return lambda X, Y, dense_output: pairwise(X, Y, dense_output, symmetric)


def get_feature_indices(rng, n_sub_features, n_features, n_components):
    arange = np.arange(n_features)
    mat = np.array(
        [rng.choice(arange, size=n_sub_features, replace=False)
         for _ in range(n_components)]
    )
    return mat.ravel()


class RandomKernel(BaseEstimator, TransformerMixin):
    """Approximates feature map of the ANOVA/all-subsets kernel by Monte Carlo
    approximation by Random Kernel Feature map.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    kernel : str (default="anova")
        Kernel to be approximated.
        "anova", "anova_cython", "all-subsets", "dot", or "pairwise"
        can be used.

    degree : int (default=2)
        Parameter of the ANOVA kernel.

    distribution : str, (default="rademacher")
        Distribution for random_weights_.
        "rademacher", "gaussian", "laplace", "uniform", or "sparse_rademacher"
        can be used.

    dense_output : bool (default=True)
        Whether randomized feature matrix is dense or sparse.
        For kernel='anova', if dense_output = False,
        distribution='sparse_rademacher', and X is sparse matrix, output random
        feature matrix will become sparse matrix.
        For kernel='anova_cython', if dense_output=False, output random feature
        matrix will become sparse matrix.

    p_sparse : float (default=0.)
        Sparsity parameter for "sparse_rademacher" distribution.
        If p_sparse = 0, "sparse_rademacher" is equivalent to "rademacher".

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_weights_ : array, shape (n_features, n_components)
        The sampled basis.

    References
    ----------
    [1] Random Feature Maps for the Itemset Kernel.
    Kyohei Atarashi, Subhransu Maji, and Satoshi Oyama
    In AAAI 2019.
    (https://www.aaai.org/ojs/index.php/AAAI/article/view/4188)
    """
    def __init__(self, n_components=100, kernel='anova', degree=2,
                 distribution='rademacher', dense_output=True, p_sparse=0.,
                 random_state=None):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.distribution = distribution
        self.dense_output = dense_output
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
        distribution = self.distribution.lower()
        self.random_weights_ = get_random_matrix(random_state, distribution,
                                                 size, self.p_sparse)
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
        X = check_array(X, accept_sparse=['csr'])
        if isinstance(self.kernel, str):
            if self.kernel == 'anova':
                kernel_ = _anova(self.degree)
            elif self.kernel == 'anova_cython':
                kernel_ = _anova_fast(self.degree)
            elif self.kernel == 'all_subsets':
                kernel_ = all_subsets
            elif self.kernel == 'dot':
                kernel_ = dot()
            else:
                raise ValueError('Kernel {} is not supported. '
                                 'Use "anova", "anova_cython", "all_subsets", '
                                 '"dot", or "pairwise".'
                                 .format(self.kernel))
        else:
            kernel_ = self.kernel
        dense_output = self.dense_output
        # for sparse output
        if not dense_output:
            if not (issparse(self.random_weights_) and issparse(X)):
                warnings.warn("dense_output=False is valid only when both "
                              "X and random_weights_ are sparse. "
                              "dense_output is changed to True now.")
                dense_output = True

        output = kernel_(X, self.random_weights_.T, dense_output)
        output /= sqrt(self.n_components)
        return output


class RandomSubsetKernel(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, n_sub_features=5, kernel='anova',
                 degree=2, distribution='rademacher', symmetric=False,
                 dense_output=False, random_state=None):
        self.n_components = n_components
        self.n_sub_features = n_sub_features
        self.degree = degree
        self.kernel = kernel
        self.distribution = distribution
        self.symmetric = symmetric
        self.dense_output = dense_output
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
        if self.kernel not in ['anova', 'anova_cython']:
            raise ValueError("RandomSubsetKernel now does not support"
                             " {} kernel.".format(self.kernel))

        if self.n_sub_features < self.degree:
            raise ValueError("n_sub_features < degree.")

        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        size = (self.n_sub_features * self.n_components, )
        distribution = self.distribution.lower()
        data = get_random_matrix(random_state, distribution, size=size)
        col = np.repeat(np.arange(self.n_components), self.n_sub_features)
        row = get_subfeatures_indices(self.n_components, n_features,
                                      self.n_sub_features, random_state)
        shape = (n_features, self.n_components)
        self.random_weights_ = csr_matrix((data, (row, col)),
                                          shape=shape)
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

        if isinstance(self.kernel, str):
            if self.kernel == 'anova':
                kernel_ = _anova(self.degree)
            elif self.kernel == 'anova_cython':
                kernel_ = _anova_fast(self.degree)
            elif self.kernel == 'all_subsets':
                kernel_ = all_subsets
            elif self.kernel == 'dot':
                kernel_ = dot()
            else:
                raise ValueError('Kernel {} is not supported. '
                                 'Use "anova", "anova_cython", "all_subsets", '
                                 '"dot", or "pairwise".'
                                 .format(self.kernel))
        else:
            kernel_ = self.kernel
        dense_output = self.dense_output
        # for sparse output
        if not dense_output:
            if not (issparse(self.random_weights_) and issparse(X)):
                warnings.warn("dense_output=False is valid only when both "
                              "X and random_weights_ are sparse. "
                              "dense_output is changed to True now.")
                dense_output = True

        const = comb(n_features, self.degree)
        const /= comb(self.n_sub_features, self.degree)
        output = kernel_(X, self.random_weights_.T, dense_output)
        output /= sqrt(self.n_components)
        output *= sqrt(const)
        return output
