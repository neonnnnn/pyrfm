# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin
from sklearn.kernel_approximation import RBFSampler
from .adagrad_fast import _adagrad_fast
from ..dataset_fast import get_dataset
from ..random_feature.random_features_fast import get_fast_random_feature


class BaseAdaGradEstimator(BaseLinear):
    LOSSES = {
        'squared': Squared(),
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge(),
        'log': Logistic()
    }

    stochastic = True

    def __init__(self, transformer=RBFSampler(), eta0=1.0, loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0, intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 eps=1e-6, warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        self.transformer = transformer
        self.eta0 = eta0
        self.loss = loss
        self.C = C
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.intercept_decay = intercept_decay
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.fast_solver = fast_solver
        self.shuffle = shuffle

    def _init_params(self, X, y):
        super(BaseAdaGradEstimator, self)._init_params(X, y)
        n_components = self.transformer.n_components
        if not (self.warm_start and hasattr(self, 'acc_grad_')):
            self.acc_grad_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'acc_grad_norm_')):
            self.acc_grad_norm_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'acc_grad_intercept_')):
            self.acc_grad_intercept_ = np.zeros(self.intercept_.shape)

        if not (self.warm_start
                and hasattr(self, 'acc_grad_norm_intercept_')):
            self.acc_grad_norm_intercept_ = np.zeros(self.intercept_.shape)

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : classifier
            Returns self.
        """

        X, y = self._check_X_y(X, y, accept_sparse=['csr'])
        n_samples, n_features = X.shape
        n_components = self.transformer.n_components
        # valid hyper parameters and init parameters
        self._valid_params()
        self._init_params(X, y)

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        intercept_decay = self.intercept_decay / self.C
        random_state = check_random_state(self.random_state)
        is_sparse = sparse.issparse(X)

        it = _adagrad_fast(self.coef_, self.intercept_,
                           get_dataset(X, order='c'), X, y, self.acc_grad_,
                           self.acc_grad_norm_,  self.acc_grad_intercept_,
                           self.acc_grad_norm_intercept_, self.mean_, self.var_,
                           loss, alpha, self.l1_ratio, intercept_decay,
                           self.eta0, self.t_, self.max_iter, self.tol,
                           self.eps, is_sparse, self.verbose,
                           self.fit_intercept, self.shuffle,
                           random_state, self.transformer,
                           get_fast_random_feature(self.transformer))
        self.t_ += n_samples*(it+1)

        return self


class AdaGradClassifier(BaseAdaGradEstimator, LinearClassifierMixin):
    """AdaGrad solver for linear classifier with random feature maps.

    Random feature mapping is computed just before computing prediction and
    gradient. Minimize

    .. math::

        \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object (default=RBFSampler())
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X).

    eta0 : double (default=0.1)
        Step-size parameter.

    loss : str (default="squared_hinge")
        Which loss function to use. Following losses can be used:

        - 'squared_hinge'

        - 'hinge'

        - 'logistic'

    C : double (default=1.0)
        Weight of the loss term.

    alpha : double (default=1.0)
        Weight of the penalty term.

    l1_ratio : double (default=0)
        Ratio of L1 regularizer.
        Weight of L1 regularizer is alpha * l1_ratio and that of L2 regularizer
        is 0.5 * alpha * (1-l1_ratio).

        - l1_ratio = 0 : Ridge.

        - l1_ratio = 1 : Lasso.

        - Otherwise : Elastic Net.

    intercept_decay : double (default=0.1)
        Weight of the penalty term for intercept.

    normalize : bool (default=False)
        Whether normalize random features or not.
        If true, the adagrad solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool (default=True)
        Whether to fit intercept (bias term) or not.

    max_iter : int (default=100)
        Maximum number of iterations.

    tol : double (default=1e-6)
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the AdaGrad solver stops learning.

    eps : double (default=1e-4)
        A small double to avoid zero-division.

    warm_start : bool (default=False)
        Whether to activate warm-start or not.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool (default=True)
        Verbose mode or not.

    fast_solver : bool (default=True)
        Use cython fast solver or not. This argument is valid when transformer
        is implemented in random_features_fast.pyx/pxd

    shuffle : bool (default=True)
        Whether to shuffle data before each epoch or not.

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
        The sum of gradients and sum of norm of gradient for intercept.
        They are used in adagrad.

    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    self.t_ : int
        The number of iteration.

    References
    ----------
    [1] Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization.
    Jonh Duchi, Elad Hazan, and Yoram Singer.
    JMLR 2011 (vol 12), pp. 2121--2159.

    """

    LOSSES = {
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge(),
        'log': Logistic()
    }

    def __init__(self, transformer=RBFSampler(), eta0=0.1, loss='squared_hinge',
                 C=1.0, alpha=1.0, l1_ratio=0., intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 eps=1e-4, warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(AdaGradClassifier, self).__init__(
            transformer, eta0, loss, C, alpha, l1_ratio, intercept_decay,
            normalize, fit_intercept, max_iter, tol, eps, warm_start,
            random_state, verbose, fast_solver, shuffle
        )


class AdaGradRegressor(BaseAdaGradEstimator, LinearRegressorMixin):
    """AdaGrad solver for linear regression with random feature maps.

    Random feature mapping is computed just before computing prediction and
    gradient. Minimize
    
    .. math::

        \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object (default=RBFSampler())
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X).

    eta0 : double (default=0.1)
        Step-size parameter.

    loss : str (default="squared")
        Which loss function to use. Following losses can be used:

        - 'squared'

    C : double (default=1.0)
        Weight of the loss term.

    alpha : double (default=1.0)
        Weight of the penalty term.

    l1_ratio : double (default=0)
        Ratio of L1 regularizer.
        Weight of L1 regularizer is alpha * l1_ratio and that of L2 regularizer
        is 0.5 * alpha * (1-l1_ratio).

        - l1_ratio = 0 : Ridge.

        - If l1_ratio = 1 : Lasso.

        - Elastic Net.

    intercept_decay : double (default=0.1)
        Weight of the penalty term for intercept.

    normalize : bool (default=False)
        Whether normalize random features or not.
        If true, the adagrad solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool (default=True)
        Whether to fit intercept (bias term) or not.

    max_iter : int (default=100)
        Maximum number of iterations.

    tol : double (default=1e-6)
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the AdaGrad solver stops learning.

    eps : double (default=1e-4)
        A small double to avoid zero-division.

    warm_start : bool (default=False)
        Whether to activate warm-start or not.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool (default=True)
        Verbose mode or not.

    fast_solver : bool (default=True)
        Use cython fast solver or not. This argument is valid when transformer
        is implemented in random_features_fast.pyx/pxd

    shuffle : bool (default=True)
        Whether to shuffle data before each epoch or not.

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
        The sum of gradients and sum of norm of gradient for intercept.
        They are used in adagrad.

    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    self.t_ : int
        The number of iteration.

    References
    ----------
    [1] Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization.
    Jonh Duchi, Elad Hazan, and Yoram Singer.
    JMLR 2011 (vol 12), pp. 2121--2159.

    """

    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, transformer=RBFSampler(), eta0=0.1, loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0., intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 eps=1e-4, warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(AdaGradRegressor, self).__init__(
            transformer, eta0, loss, C, alpha, l1_ratio, intercept_decay,
            normalize, fit_intercept, max_iter, tol, eps, warm_start,
            random_state, verbose, fast_solver, shuffle
        )
