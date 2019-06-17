from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from scipy import sparse


from sklearn.utils.extmath import safe_sparse_dot, row_norms
from sklearn.utils import check_X_y, check_random_state
from lightning.impl.dataset_fast import get_dataset

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from ..maji_berg import SparseMB
from .cd_primal_sparse_mb import _cd_primal_epoch
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin


class BaseSparseMBEstimator(BaseLinear):
    """Linear model with feature map approximating the intersection (min)
    kernel by sparse explicit feature map, which is proposed by S.Maji
    and A.C.Berg. SparseMB does not approximate min kernel only itself.
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
    LOSSES = {
        'squared': Squared(),
        'suqared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, n_components=1000, loss='squared_hinge', penalty='l2',
                 solver='cd', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-3, warm_start=False,
                 random_state=None, verbose=True):
        self.n_components = n_components
        self.loss = loss
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        if not self.warm_start and hasattr(self, 'transformer_'):
            self.transformer_ = SparseMB(n_components=self.n_components)
            self.transformer_.fit(X)
        X = self.transformer_.transform(X)

        n_samples, n_features = X.shape
        if not self.warm_start and hasattr(self, 'coef_'):
            self.coef_ = np.zeros(n_features)

        if not self.warm_start and hasattr(self, 'intercept_'):
            self.intercept_ = 0.

        if self.loss not in self.LOSSES:
            raise ValueError("loss {} is not supported.")

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C

        # make tridiagonal matrix
        H = sparse.diags([-1, 2+self.eps, -1], [-1, 0, 1],
                         shape=(self.n_components, self.n_components))
        H[0, 0] = 1+self.eps
        H[self.n_components-1, self.n_components-1] = 1+self.eps

        y_pred = self.decision_function(X)
        X_col_norms = row_norms(X.T, True)
        random_state = check_random_state(self.random_state)
        _cd_primal_epoch(self.coef_, self.intercept_, get_dataset(X, 'f'), y,
                         X_col_norms, y_pred, get_dataset(H, 'c'), alpha, loss,
                         self.max_iter, self.tol, self.fit_intercept,
                         random_state, self.verbose)


class SparseMBClassifier(BaseSparseMBEstimator, LinearClassifierMixin):
    LOSSES = {
        'suqared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, n_components=1000, loss='squared_hinge', penalty='l2',
                 solver='cd', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-4, warm_start=False,
                 random_state=None, verbose=True):
        super(SparseMBClassifier, self).__init__(
            n_components, loss, penalty, solver, C, alpha, fit_intercept,
            max_iter, tol, eps, warm_start, random_state, verbose
        )

    def _check_X_y(self, X, y):
        return check_X_y(X, y, True, multi_output=False, y_numeric=False)


class SparseMBRegressor(BaseSparseMBEstimator, LinearRegressorMixin):
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, n_components=1000, loss='squared', penalty='l2',
                 solver='cd', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-4, warm_start=False,
                 random_state=None, verbose=True):
        super(SparseMBRegressor, self).__init__(
            n_components, loss, penalty, solver, C, alpha, fit_intercept,
            max_iter, tol, eps, warm_start, random_state, verbose
        )

    def _check_X_y(self, X, y):
        return check_X_y(X, y, True, multi_output=False, y_numeric=True)