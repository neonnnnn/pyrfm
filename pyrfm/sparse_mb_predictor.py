from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from scipy import sparse


from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import compute_sample_weight
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from sklearn.exceptions import ConvergenceWarning
from .loss import Squared, SquaredHinge, Logistic
from .maji_berg import SparseMB


class BaseSparseMBEstimator(BaseEstimator):
    LOSSES = ["squared_hinge", "hinge", "logistic", "squared"]

    def __init__(self, loss='squared_hinge', penalty='l2', solver='cd', C=1.0,
                 alpha=1.0, fit_intercept=True, max_iter=100, tol=1e-6,
                 warm_start=False, random_state=None, n_components=1000):
        self.loss = loss
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.n_components = n_components

    def decision_function(self, X):
        pred = safe_sparse_dot(self.mb_transformer_.transform(X), self.coef_.T)
        if self.fit_intercept:
            pred += self.intercept_
        return pred

    def fit(self, X, y):
        X, y = check_X_y(X, y, True)
        if not self.warm_start and hasattr(self, 'mb_transformer_'):
            self.mb_transformer_ = SparseMB(n_components=self.n_components)
            self.mb_transformer_.fit(X)
        X = self.mb_transformer_.transform(X)

        n_samples, n_features = X.shape
        if not self.warm_start and hasattr(self, 'coef_'):
            self.coef_ = np.zeros(n_features)

        if not self.warm_start and hasattr(self, 'intercept_'):
            self.intercept_ = 0.

        if not self.loss in self.LOSSES:
            raise ValueError("loss {} is not supported.")

    def predict(self, X):
        pred = self.decision_function(X)
        out = self.label_binarizer_.inverse_transform(pred)
        return out


class SparseMBClassifier(BaseSparseMBEstimator, ClassifierMixin):
    LOSSES = ["squared_hinge", "hinge", "logistic"]

    def __init__(self, loss='squared_hinge', penalty='l2', solver='cd', C=1.0,
                 alpha=1.0, fit_intercept=True, max_iter=100, tol=1e-6,
                 warm_start=False, random_state=None, n_components=1000):
        super(SparseMBClassifier, self).__init__(
            loss, penalty, solver, C, alpha, fit_intercept, max_iter, tol,
            warm_start, random_state, n_components
        )


class SparseMBRegressor(BaseSparseMBEstimator, ClassifierMixin):
    LOSSES = ["squared"]

    def __init__(self, loss='squared', penalty='l2', solver='cd', C=1.0,
                 alpha=1.0, fit_intercept=True, max_iter=100, tol=1e-6,
                 warm_start=False, random_state=None, n_components=1000):
        super(SparseMBRegressor, self).__init__(
            loss, penalty, solver, C, alpha, fit_intercept, max_iter, tol,
            warm_start, random_state, n_components
        )