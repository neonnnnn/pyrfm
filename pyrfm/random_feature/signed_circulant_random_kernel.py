# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from ..kernels import safe_power
from math import sqrt
from scipy.sparse import issparse
from .signed_circulant_random_projection import SignedCirculantRandomMatrix


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

    Attributes
    ----------
    transformer_ : SignedCirculantRandomMatrix
        Transformer object of signed circulant random matrix.

    References
    ----------
    [1] Random Feature Maps for the Itemset Kernel.
    Kyohei Atarashi, Subhransu Maji, and Satoshi Oyama.
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
        self.transformer_ = SignedCirculantRandomMatrix(
            self.n_components, distribution='rademacher', random_fourier=False,
            random_state=random_state
        )
        self.transformer_.fit(X)
        self.n_components = self.transformer_.n_components
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
        check_is_fitted(self.transformer_, "random_weights_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        Ds = [1.]
        # O(mND \log d) (m=degree, N=n_samples, D=n_components, d=n_features)
        for deg in range(1, self.degree+1):
            if deg % 2 == 0:
                # O(nnz(X)) ( = O(Nd))
                # D.shape = (n_samples, 1)
                if issparse(X):
                    D = np.array(np.sum(safe_power(X, 2), axis=1))
                else:
                    D = np.sum(safe_power(X, 2), axis=1, keepdims=True)
            else:
                # O(ND\log d)
                # D.shape = (n_samples, n_components)
                D = self.transformer_.transform(safe_power(X, deg))
                D *= sqrt(self.n_components)
            Ds.append(D)

        # O(m^2ND)
        As = [1, self.transformer_.transform(X)]
        for deg in range(2, self.degree+1):
            A = np.zeros((n_samples, self.n_components))
            for t in range(1, deg+1):
                A += (-1) ** (t+1) * As[deg - t] * Ds[t]
            As.append(A / deg)
        return As[-1] / sqrt(self.n_components)
