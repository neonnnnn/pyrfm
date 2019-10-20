# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from scipy.sparse import csc_matrix
import warnings
from ..dataset_fast import get_dataset
from .unarize import unarize, make_sparse_mb


class MB(BaseEstimator, TransformerMixin):
    """Approximates feature map of the intersection (min) kernel by explicit
    feature map, which is proposed by S.Maji and A.C.Berg.

    Parameters
    ----------
    n_components : int (default=1000)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    References
    ----------
    [1] Max-Margin Additive Classifiers for Detection
    Subhransu Maji, Alexander C. Berg.
    In ICCV 2009.
    (http://acberg.com/papers/mb09iccv.pdf)

    """
    def __init__(self, n_components=1000):
        self.n_components = n_components

    def fit(self, X, y=None):
        """Compute the number of grids according to n_features.

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
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        self.n_grids_ = self.n_components // n_features
        n_components = self.n_grids_ * n_features
        if self.n_components % n_features != 0:
            warnings.warn("self.n_components is indivisible by n_features."
                          "n_components is changed from {} to {}"
                          .format(self.n_components, n_components))
            self.n_components = n_components
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
        check_is_fitted(self, "n_grids_")
        X = check_array(X, accept_sparse=['csc'])
        n_samples, n_features = X.shape
        if np.max(X) > 1:
            raise ValueError("The maximum value of X is bigger than 1.")

        if np.min(X) < 0:
            raise ValueError("The minimum value of X is lower than 1.")

        if self.n_components != self.n_grids_*n_features:
            raise ValueError("X.shape[1] is different from X_train.shape[1].")
        output = np.zeros((n_samples, self.n_components))
        unarize(output, get_dataset(X, order='c'), self.n_grids_)

        return output


class SparseMB(BaseEstimator, TransformerMixin):
    """Approximates feature map of the intersection (min) kernel by sparse
    explicit feature map, which is proposed by S.Maji and A.C.Berg.
    SparseMB does not approximate min kernel only itself.
    Linear classifier with SparseMB approximates linear classifier with MB.
    For more detail, see [1].

    Parameters
    ----------
    n_components : int (default=1000)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    References
    ----------
    [1] Max-Margin Additive Classifiers for Detection
    Subhransu Maji, Alexander C. Berg.
    In ICCV 2009.
    (http://acberg.com/papers/mb09iccv.pdf)
    """
    def __init__(self, n_components=1000):
        self.n_components = n_components

    def fit(self, X, y=None):
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
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        if self.n_components < n_features:
            raise ValueError("self.n_components is lower than n_features "
                             "(X.shape[1]).")
        self.n_grids_ = self.n_components // n_features
        n_components = self.n_grids_*n_features

        if self.n_components % n_features != 0:
            warnings.warn("self.n_components is indivisible by n_features."
                          "n_components is changed from {} to {}"
                          .format(self.n_components, n_components))
            self.n_components = n_components
        return self

    def transform(self, X):
        """Apply the approximate feature map to X for SparseMBEstimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "n_grids_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape

        if np.max(X) > 1:
            raise ValueError("The maximum value of X is bigger than 1.")

        if np.min(X) < 0:
            raise ValueError("The minimum value of X is lower than 1.")

        if self.n_components != self.n_grids_*n_features:
            raise ValueError("X.shape[1] is different from X_train.shape[1].")
        dataset = get_dataset(X, order='c')
        data = np.zeros(X.size*2)
        row = np.zeros(X.size*2, dtype=np.int32)
        col = np.zeros(X.size*2, dtype=np.int32)

        make_sparse_mb(data, row, col, dataset, self.n_grids_)
        return csc_matrix((data, (row, col)),
                          shape=(n_samples, self.n_components))
