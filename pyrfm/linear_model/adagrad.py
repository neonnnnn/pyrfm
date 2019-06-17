from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from scipy import sparse


from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_X_y, check_random_state

from .loss import Squared, SquaredHinge, Logistic, Hinge
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin
from sklearn.kernel_approximation import RBFSampler


class BaseAdaGradEstimator(BaseLinear):
    """AdaGrad solver for linear models with random feature maps.
    Random feature mapping is computed just before computing prediction for
    each sample.
    """
    LOSSES = {
        'squared': Squared(),
        'suqared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, transformer=RBFSampler(), eta=1.0, loss='squared_hinge',
                 penalty='l2', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6,  eps=1e-6, warm_start=False,
                 random_state=None, verbose=True):
        self.transformer = transformer
        self.eta = eta
        self.loss = loss
        self.penalty = penalty
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
        if not self.warm_start:
            self.transformer.fit(X)

        n_samples, n_features = X.shape
        if not hasattr(self.transformer, 'n_components_actual_'):
            n_components = self.transformer.n_components
        else:
            n_components = self.transformer.n_components_actual_

        if not self.warm_start and hasattr(self, 'coef_'):
            self.coef_ = np.zeros(n_components)

        if not self.warm_start and hasattr(self, 'intercept_'):
            self.intercept_ = 0.

        if not self.warm_start and hasattr(self, 'acc_grad_'):
            self.acc_grad_ = np.zeros(n_components)

        if not self.warm_start and hasattr(self, 'acc_grad_norm_'):
            self.acc_grad_norm_ = np.zeros(n_components)

        if self.fit_intercept:
            if not self.warm_start and hasattr(self, 'acc_grad_intercept_'):
                self.acc_grad_intercept_ = 0.

            if not self.warm_start and hasattr(self, 'acc_grad_norm_intercept_'):
                self.acc_grad_norm_intercept_ = 0.

        if not self.warm_start and hasattr(self, 't_'):
            self.t_ = 1

        if self.loss not in self.LOSSES:
            raise ValueError("loss {} is not supported.")

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        random_state = check_random_state(self.random_state)

        is_sparse = sparse.issparse(X)
        for it in range(self.max_iter):
            viol = 0
            for i in random_state.permutation(n_samples):
                if is_sparse:
                    x = self.transformer.transform(X[i])
                else:
                    x = self.transformer.transform(np.atleast_2d(X[i]))
                y_pred = safe_sparse_dot(x, self.coef_, True)
                if self.fit_intercept:
                    y_pred += self.intercept_

                dloss = loss.dloss(y_pred, y[i])
                if dloss != 0:
                    grad = dloss * x.ravel()
                    self.acc_grad_ += grad
                    self.acc_grad_norm_ += grad**2
                eta_t = self.eta * self.t_
                denom = np.sqrt(self.acc_grad_norm_) + self.eps
                denom += alpha * eta_t
                coef_new = eta_t * (-self.acc_grad_ / self.t_) / denom
                viol += np.sum(np.abs(coef_new-self.coef_))
                self.coef_ = coef_new

                if self.fit_intercept:
                    self.acc_grad_intercept_ += dloss
                    self.acc_grad_norm_ += dloss*dloss
                    denom = np.sqrt(self.acc_grad_norm_intercept_) + self.eps
                    intercept_new = - eta_t * self.acc_grad_norm_intercept_
                    intercept_new /= self.t_ * denom
                    viol += np.abs(intercept_new - self.intercept_)
                    self.intercept_ = intercept_new

                self.t_ += 1

            if self.verbose:
                print("Iteration {} Violation {}".format(it, viol))

            if viol < self.tol:
                if self.verbose:
                    print("Converged at iteration {}".format(it))
                break
        return self


class AdaGradClassifier(BaseAdaGradEstimator, LinearClassifierMixin):
    LOSSES = {
        'suqared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, transformer=RBFSampler(), eta=1.0, loss='squared_hinge',
                 penalty='l2', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-4, warm_start=False,
                 random_state=None, verbose=True):
        super(AdaGradClassifier, self).__init__(
            transformer, eta, loss, penalty, C, alpha, fit_intercept,
            max_iter, tol, eps, warm_start, random_state, verbose
        )

    def _check_X_y(self, X, y):
        return check_X_y(X, y, True, multi_output=False, y_numeric=False)


class AdaGradRegressor(BaseAdaGradEstimator, LinearRegressorMixin):
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, transformer=RBFSampler(), eta=1.0, loss='squared',
                 penalty='l2', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-4, warm_start=False,
                 random_state=None, verbose=True):
        super(AdaGradRegressor, self).__init__(
            transformer, eta, loss, penalty, C, alpha, fit_intercept,
            max_iter, tol, eps, warm_start, random_state, verbose
        )

    def _check_X_y(self, X, y):
        return check_X_y(X, y, True, multi_output=False, y_numeric=True)
