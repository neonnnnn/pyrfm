import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from scipy.sparse import csr_matrix
from math import sqrt
import warnings
from lightning.impl.dataset_fast import get_dataset
from .unarize import unarize


class MB(BaseEstimator, TransformerMixin):
    """Approximates feature map of the intersection (min) kernel by explicit
    feature map, which is proposed by S.Maji and A.C.Berg.

    Parameters
    ----------
    n_components : int
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
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        self.n_grids_ = self.n_components // n_features
        self.n_components_actual_ = self.n_grids_ * n_features
        if self.n_components % n_features != 0:
            warnings.warn("self.n_components is indivisible by n_features."
                          "Output.shape[1] is "
                          "{}".format(self.n_components_actual_))

        return self

    def transform(self, X):
        check_is_fitted(self, "n_components_actual_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        if np.max(X) > 1:
            raise ValueError("The maximum value of X is bigger than 1.")

        if np.min(X) < 0:
            raise ValueError("The minimum value of X is lower than 1.")

        if self.n_components_actual_ != self.n_grids_*n_features:
            raise ValueError("X.shape[1] is different from X_train.shape[1].")
        output = np.zeros((n_samples, self.n_components_actual_))
        unarize(output, get_dataset(X, order='c'), self.n_grids_)

        return output


class SparseMB(BaseEstimator, TransformerMixin):
    """Approximates feature map of the intersection (min) kernel by sparse
    explicit feature map, which is proposed by S.Maji and A.C.Berg.
    SparseMB does not approximate min kernel only itselt.
    Linear classifier with SparseMB approximates linear classifier with MB.
    For more detail, see [1].

    Parameters
    ----------
    n_components : int
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    References
    ----------
    [1] Max-Margin Additive Classifiers for Detection
    Subhransu Maji, Alexander C. Berg.
    In ICCV 2009.
    (http://acberg.com/papers/mb09iccv.pdf)
    """
    def __init__(self, n_components=100):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        self.n_grids_ = self.n_components // n_features
        self.n_components_actual_ = self.n_grids_*n_features
        if self.n_components < n_features:
            raise ValueError("self.n_components is lower than n_features "
                             "(X.shape[1]).")
        if self.n_components % n_features != 0:
            warnings.warn("self.n_components is indivisible by n_features."
                          "Output.shape[1] is "
                          "{}".format(self.n_components_actual_))
        return self

    def transform(self, X):
        check_is_fitted(self, "n_components_actual_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        if np.max(X) > 1:
            raise ValueError("The maximum value of X is bigger than 1.")

        if np.min(X) < 0:
            raise ValueError("The minimum value of X is lower than 1.")

        if self.n_components_actual_ != self.n_grids_*n_features:
            raise ValueError("X.shape[1] is different from X_train.shape[1].")
        row = np.repeat(np.arange(n_samples), 2*n_features)
        row = row.astype(np.int64)
        col = np.array([np.ceil(X*self.n_grids_),
                        np.ceil(X*self.n_grids_+1)])
        col = col.T.ravel()
        col = col.astype(np.int64)
        X *= self.n_grids_
        X -= np.ceil(X)
        data = np.array([(1.-X).ravel(), X.ravel()]).T.ravel() / sqrt(self.n_grids_)
        return csr_matrix((data, (row, col)))
