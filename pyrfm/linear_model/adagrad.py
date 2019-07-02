import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_X_y, check_random_state

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin
from sklearn.kernel_approximation import RBFSampler
from .adagrad_fast import _adagrad_fast
from sklearn.utils.validation import check_is_fitted
from lightning.impl.dataset_fast import get_dataset


class BaseAdaGradEstimator(BaseLinear):
    """AdaGrad solver for linear models with random feature maps.
    Random feature mapping is computed just before computing prediction and
    gradient.
    minimize  \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object, default=RBFSampler()
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X).

    eta : double, default=1.0
        Step-size parameter.

    loss : str
        Which loss function to use. Following losses can be used:
            'squared' (for regression)
            'squared_hinge' (for classification)
            'hinge' (for classification)
            'logistic' (for classification)

    C : double, default=1.0
        Weight of the loss term.

    alpha : double, default=1.0
        Weight of the penalty term.

    l1_ratio : double, default=0
        Ratio of L1 regularizer.
        Weight of L1 regularizer is alpha * l1_ratio and that of L2 regularizer
        is 0.5 * alpha * (1-l1_ratio).
        If l1_ratio = 0 : Ridge.
        else If l1_ratio = 1 : Lasso.
        else : Elastic Net.

    normalize : bool, default=False
        Whether normalize random features or not.
        If true, the adagrad solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool, default=True
        Whether to fit intercept (bias term) or not.

    max_iter : int
        Maximum number of iterations.

    tol : double
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the AdaGrad solver stops learning.

    eps : double
        A small double to avoid zero-division.

    warm_start : bool
        Whether to activate warm-start or not.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, default=True
        Verbose mode or not.

    fast_solver : bool, default=True
        Use cython fast solver or not. This argument is valid when transformer
        is in {RandomFourier|RandomMaclaurin|TensorSketch|RandomKernel}.

    Attributes
    ----------
    self.coef_ : array, shape (n_components, )
        The learned coefficients of the linear model.

    self.intercept_ : array, shape (1, )
        The learned intercept (bias) of the linear model.

    self.acc_grad_, self.acc_grad_norm_ : array, shape (n_components, )
        The sum of gradients and sum of norm of gradient for coefficients.
        They are used in adagrad.

    self.acc_grad_intercept_, self.acc_grad_intercept_norm :
     array, shape (n_components, )
        The sum of gradients and sum of norm of gradient for intercept_.
        They are used in adagrad.

    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    References
    ---------
    [1] Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization.
    Jonh Duchi, Elad Hazan, and Yoram Singer.
    JMLR 2011 (vol 12), pp. 2121--2159.
    """
    LOSSES = {
        'squared': Squared(),
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    stochastic = True

    def __init__(self, transformer=RBFSampler(), eta=1.0, loss='squared_hinge',
                 C=1.0, alpha=1.0, l1_ratio=0, normalize=False,
                 fit_intercept=True, max_iter=100, tol=1e-6,  eps=1e-6,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True):
        self.transformer = transformer
        self.transformer_ = transformer
        self.eta = eta
        self.loss = loss
        self.C = C
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.fast_solver = fast_solver

    def _predict(self, X):
        check_is_fitted(self, "coef_")
        y_pred = np.zeros((X.shape[0], ))
        is_sparse = sparse.issparse(X)
        for i, xi in enumerate(X):
            if is_sparse:
                xi_trans = self.transformer.transform(xi).ravel()
            else:
                xi_trans = self.transformer.transform(np.atleast_2d(xi)).ravel()

            if self.normalize:
                xi_trans = (xi_trans - self.mean_) / np.sqrt(self.var_)
            y_pred[i] = safe_sparse_dot(xi_trans, self.coef_)
            y_pred[i] += self.intercept_

        return y_pred

    def fit(self, X, y):
        X, y = self._check_X_y(X, y, accept_sparse=['csr'])
        if not self.warm_start:
            self.transformer.fit(X)

        n_samples, n_features = X.shape
        if not (hasattr(self.transformer, 'n_components_actual_')):
            n_components = self.transformer.n_components
        else:
            n_components = self.transformer.n_components_actual_

        if not (self.warm_start and hasattr(self, 'coef_')):
            self.coef_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'intercept_')):
            self.intercept_ = np.zeros((1,) )

        if not (self.warm_start and hasattr(self, 'acc_grad_')):
            self.acc_grad_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'acc_grad_norm_')):
            self.acc_grad_norm_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'acc_grad_intercept_')):
            self.acc_grad_intercept_ = np.zeros(self.intercept_.shape)

        if not (self.warm_start
                and hasattr(self, 'acc_grad_norm_intercept_')):
            self.acc_grad_norm_intercept_ = np.zeros(self.intercept_.shape)

        if not (self.warm_start and hasattr(self, 't_')):
            self.t_ = 1

        if self.loss not in self.LOSSES:
            raise ValueError("loss {} is not supported.".format(self.loss))

        if self.normalize:
            if not (self.warm_start and hasattr(self, 'mean_')):
                self.mean_ = np.zeros((n_components, ))

            if not (self.warm_start and hasattr(self, 'var_')):
                self.var_ = np.zeros((n_components,))
        else:
            self.mean_ = None
            self.var_ = None
        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        random_state = check_random_state(self.random_state)

        id_transformer = self._get_id_transformer()
        if not self.fast_solver:
            id_transformer = -1
        params = self._get_transformer_params(id_transformer)

        is_sparse = sparse.issparse(X)
        it = _adagrad_fast(self.coef_, self.intercept_,
                           get_dataset(X, order='c'), X, y, self.acc_grad_,
                           self.acc_grad_norm_,  self.acc_grad_intercept_,
                           self.acc_grad_norm_intercept_, self.mean_, self.var_,
                           loss, alpha, self.l1_ratio, self.eta, self.t_,
                           self.max_iter, self.tol, self.eps, 1e-6, is_sparse,
                           self.verbose, self.fit_intercept, random_state,
                           self.transformer, id_transformer, **params)
        self.t_ += n_samples*(it+1)

        return self


class AdaGradClassifier(BaseAdaGradEstimator, LinearClassifierMixin):
    LOSSES = {
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, transformer=RBFSampler(), eta=1.0, loss='squared_hinge',
                 C=1.0, alpha=1.0, l1_ratio=0., normalize=False,
                 fit_intercept=True, max_iter=100, tol=1e-6, eps=1e-4,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True):
        super(AdaGradClassifier, self).__init__(
            transformer, eta, loss, C, alpha, l1_ratio, normalize,
            fit_intercept, max_iter, tol, eps, warm_start, random_state,
            verbose, fast_solver
        )


class AdaGradRegressor(BaseAdaGradEstimator, LinearRegressorMixin):
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, transformer=RBFSampler(), eta=1.0, loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0., normalize=False,
                 fit_intercept=True, max_iter=100, tol=1e-6, eps=1e-4,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True):
        super(AdaGradRegressor, self).__init__(
            transformer, eta, loss, C, alpha, l1_ratio, normalize,
            fit_intercept, max_iter, tol, eps, warm_start, random_state,
            verbose, fast_solver
        )
