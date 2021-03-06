# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from scipy.special import factorial, binom
from scipy.sparse import issparse, hstack, csr_matrix
from .utils_random_fast import get_subfeatures_indices
from .utils import get_random_matrix
import warnings


class RandomMaclaurin(BaseEstimator, TransformerMixin):
    """Approximates feature map of a dot product kernel by Monte Carlo
    approximation of its Maclaurin expansion.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    p : int (default=10)
        Parameter of the distribution that determines which components of the
        Maclaurin series are approximated.

    kernel : str or callable (default="poly")
        Type of kernel function. 'poly', 'exp', or callable are accepted.
        If callable, its arguments are two numpy-like objects, and return a
        numpy-like object.
        if str, only 'poly' or 'exp' is acceptable.

    degree : int (default=2)
        Parameter of the polynomial product kernel.

    distribution : str, (default="rademacher")
        Distribution for random_weights_.
        "rademacher", "gaussian", "laplace", "uniform", or "sparse_rademacher"
        can be used.

    gamma : float or str (default="auto")
        Parameter of the exponential kernel.

    bias : float (default=0)
        Parameter of the polynomial kernel.

    coefs : list-like (default=None)
        list of coefficients of Maclaurin expansion.

    max_expansion : int (default=50)
        Threshold of Maclaurin expansion.

    h01 : bool (default=False)
        Use h01 heuristic or not. See [1].

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
    orders_ : array, shape (n_components, )
        The sampled orders of the Maclaurin expansion.
        The j-th components of random feature approximates orders_[j]-th order
        of the Maclaurin expansion.

    random_weights_ : array, shape (n_features, np.sum(orders_))
        The sampled basis.

    References
    ----------
    [1] Random Feature Maps for Dot Product Kernels.
    Purushottam Kar and Harish Karnick.
    In AISTATS 2012.
    (http://proceedings.mlr.press/v22/kar12/kar12.pdf)

    """
    def __init__(self, n_components=100, p=10, kernel='poly', degree=2,
                 distribution='rademacher', gamma='auto', bias=0., coefs=None,
                 max_expansion=50, h01=False,  dense_output=True, p_sparse=0.0,
                 random_state=None):
        self.n_components = n_components
        self.p = p
        self.gamma = gamma
        self.degree = degree
        self.distribution = distribution
        # coefs of Maclaurin series.
        # If kernel is 'poly' or 'exp', this is computed automatically.
        self.coefs = coefs
        self.bias = float(bias)
        self.kernel = kernel
        self.max_expansion = max_expansion
        self.p_choice = None
        self.h01 = h01
        self.dense_output = dense_output
        self.p_sparse = p_sparse
        self.random_state = random_state

    def _set_coefs(self, gamma):
        if self.coefs is None:
            if self.kernel == 'poly':
                self.coefs = self.bias ** np.arange(self.degree+1)[::-1]
                self.coefs *= binom(self.degree,
                                    range(self.degree+1)).astype(np.float64)
                self.coefs /= factorial(range(self.degree+1))
            elif self.kernel == 'exp':
                self.coefs = gamma ** np.arange(self.max_expansion)
                self.coefs /= factorial(range(self.max_expansion))
            else:
                raise ValueError("When using the user-specific kernel "
                                 "function, coefs must be given explicitly.")
    
    def _sample_orders(self, random_state):
        coefs = np.array(self.coefs)
        if self.h01:
            coefs[1] = 0
            coefs[0] = 0

        if self.p_choice is None:
            p_choice = (1/self.p) ** (np.arange(len(coefs)) + 1)
            if np.sum(coefs == 0.) != 0:
                p_choice[coefs == 0] = 0
            p_choice /= np.sum(p_choice)
            self.p_choice = p_choice
 
        self.orders_ = random_state.choice(len(self.p_choice),
                                           self.n_components,
                                           p=self.p_choice).astype(np.int32)

    def fit(self, X, y=None):
        """Generate random weights and orders according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape

        if isinstance(self.kernel, str):
            if self.kernel not in ['exp', 'poly']:
                raise ValueError("kernel must be {'poly'|'exp'} or callable.")
        else:
            if not callable(self.kernel):
                raise ValueError("kernel must be {'poly'|'exp'} or callable.")

        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        self._set_coefs(gamma)
        self._sample_orders(random_state)
        distribution = self.distribution.lower()
        size = (n_features, np.sum(self.orders_))
        self.random_weights_ = get_random_matrix(random_state, distribution,
                                                 size, self.p_sparse)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "random_weights_")
        from .random_features_fast import transform_all_fast
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        dense_output = self.dense_output
        if not dense_output:
            if not (issparse(self.random_weights_) and issparse(X)):
                warnings.warn("dense_output=False is valid only when both "
                              "X and random_weights_ are sparse. "
                              "dense_output is changed to True now.")
                dense_output = True
        output = transform_all_fast(X, self, dense_output)

        if self.h01:
            linear = X * np.sqrt(self.coefs[1])
            dummy = np.sqrt(self.coefs[0]) * np.ones((n_samples, 1))
            if dense_output:
                if issparse(linear):
                    output = np.hstack([linear.toarray(), output])
                else:
                    output = np.hstack([linear, output])
                output = np.hstack((dummy, output))
            else:
                output = hstack([linear, output])
                output = hstack((csr_matrix(dummy), output))

        return output

    def _remove_bases(self, indices):
        cumsum = np.append(0, np.cumsum(self.orders_))
        self.orders_ = np.delete(self.orders_, indices)
        ind = np.concatenate([[j for j in range(cumsum[i], cumsum[i+1])] for i in indices])
        self.random_weights_ = np.delete(self.random_weights_, ind, axis=1)
        self.n_components = self.orders_.shape[0]
        return True


class SubfeatureRandomMaclaurin(BaseEstimator, TransformerMixin):
    """Approximates feature map of a dot product kernel by Monte Carlo
    approximation of its Maclaurin expansion with only sub features.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    n_sub_features : int (default=5)
        Number of sub features.

    p : int (default=10)
        Parameter of the distribution that determines which components of the
        Maclaurin series are approximated.

    kernel : str or callable (default="poly")
        Type of kernel function. 'poly', 'exp', or callable are accepted.
        If callable, its arguments are two numpy-like objects, and return a
        numpy-like object.
        if str, only 'poly' or 'exp' is acceptable.

    degree : int (default=2)
        Parameter of the polynomial product kernel.

    distribution : str, (default="rademacher")
        Distribution for random_weights_.
        "rademacher", "gaussian", "laplace", "uniform", or "sparse_rademacher"
        can be used.

    gamma : float or str (default="auto")
        Parameter of the exponential kernel.

    bias : float (default=0)
        Parameter of the polynomial kernel.

    coefs : list-like (default=None)
        list of coefficients of Maclaurin expansion.

    max_expansion : int (default=50)
        Threshold of Maclaurin expansion.

    h01 : bool (default=False)
        Use h01 heuristic or not. See [1].

    dense_output : bool (default=False)
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
    orders_ : array, shape (n_components, )
        The sampled orders of the Maclaurin expansion.
        The j-th components of random feature approximates orders_[j]-th order
        of the Maclaurin expansion.

    random_weights_ : csr_matrix, shape (n_features, np.sum(orders_))
        The sampled basis.

    References
    ----------
    [1] Random Feature Maps for Dot Product Kernels.
    Purushottam Kar and Harish Karnick.
    In AISTATS 2012.
    (http://proceedings.mlr.press/v22/kar12/kar12.pdf)

    [2] Sparse Random Feature Maps for the item-multiset Kernel.
    Kyohei Atarashi, Satoshi Oyama and Masahito Kurihara.
    To appear.
    """
    def __init__(self, n_components=100, n_sub_features=5, p=10, kernel='poly',
                 degree=2, distribution='rademacher', gamma='auto', bias=0.,
                 coefs=None, max_expansion=50, h01=False,  dense_output=True,
                 p_sparse=0.0, random_state=None):
        self.n_components = n_components
        self.n_sub_features = n_sub_features
        self.p = p
        self.gamma = gamma
        self.degree = degree
        self.distribution = distribution
        # coefs of Maclaurin series.
        # If kernel is 'poly' or 'exp', this is computed automatically.
        self.coefs = coefs
        self.bias = float(bias)
        self.kernel = kernel
        self.max_expansion = max_expansion
        self.p_choice = None
        self.h01 = h01
        self.dense_output = dense_output
        self.p_sparse = p_sparse
        self.random_state = random_state

    def fit(self, X, y=None):
        """Generate random weights and orders according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape

        if self.n_sub_features > n_features:
            raise ValueError("self.n_sub_features > X.shape[1].")

        if isinstance(self.kernel, str):
            if self.kernel not in ['exp', 'poly']:
                raise ValueError("kernel must be {'poly'|'exp'} or callable.")
        else:
            if not callable(self.kernel):
                raise ValueError("kernel must be {'poly'|'exp'} or callable.")

        if self.gamma == 'auto':
            gamma_ = 1.0 / X.shape[1]
        else:
            gamma_ = self.gamma

        if self.coefs is None:
            if self.kernel == 'poly':
                self.coefs = self.bias ** np.arange(self.degree+1)[::-1]
                self.coefs *= binom(self.degree,
                                    range(self.degree+1)).astype(np.float64)
                self.coefs /= factorial(range(self.degree+1))
                coefs = self.coefs
            elif self.kernel == 'exp':
                self.coefs = gamma_ ** np.arange(self.max_expansion)
                self.coefs /= factorial(range(self.max_expansion))
                coefs = self.coefs
            else:
                raise ValueError('When using the user-specific kernel function,'
                                 'coefs must be given explicitly.')
        else:
            coefs = self.coefs

        if self.h01:
            coefs[1] = 0
            coefs[0] = 0

        if self.p_choice is None:
            p_choice = (1/self.p) ** (np.arange(len(coefs)) + 1)
            if np.sum(coefs == 0.) != 0:
                p_choice[coefs == 0] = 0
            p_choice /= np.sum(p_choice)
            self.p_choice = p_choice

        self.orders_ = random_state.choice(len(self.p_choice),
                                           self.n_components,
                                           p=self.p_choice).astype(np.int32)

        distribution = self.distribution.lower()
        size = (self.n_sub_features * np.sum(self.orders_), )
        data = get_random_matrix(random_state, distribution, size=size)
        col = np.repeat(np.arange(np.sum(self.orders_)), self.n_sub_features)
        row = get_subfeatures_indices(np.sum(self.orders_), n_features,
                                      self.n_sub_features, random_state)
        shape = (n_features, np.sum(self.orders_))
        self.random_weights_ = csr_matrix((data, (row, col)), shape=shape)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "random_weights_")
        from .random_features_fast import transform_all_fast
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        dense_output = self.dense_output
        if not dense_output:
            if not (issparse(self.random_weights_) and issparse(X)):
                warnings.warn("dense_output=False is valid only when both "
                              "X and random_weights_ are sparse. "
                              "dense_output is changed to True now.")
                dense_output = True
        output = transform_all_fast(X, self, dense_output)

        if self.h01 and self.bias != 0:
            linear = X * np.sqrt(self.degree*self.bias**(self.degree-1))
            dummy = np.sqrt(self.bias ** self.degree) * np.ones((n_samples, 1))
            if dense_output:
                if issparse(linear):
                    output = np.hstack([linear.toarray(), output])
                else:
                    output = np.hstack([linear, output])
                output = np.hstack((dummy, output))
            else:
                output = hstack([linear, output])
                output = hstack((csr_matrix(dummy), output))

        return output
