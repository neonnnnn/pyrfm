# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin
from sklearn.kernel_approximation import RBFSampler
from .sgd_fast import _sgd_fast
from ..dataset_fast import get_dataset
from ..random_feature.random_features_fast import get_fast_random_feature


class BaseSGDEstimator(BaseLinear):
    LEARNING_RATE = {
        'constant': 0,
        'pegasos': 1,
        'inv_scaling': 2,
        'optimal': 3,
    }
    LOSSES = {
        'squared': Squared(),
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge(),
        'log': Logistic()
    }
    stochastic = True

    def __init__(self, transformer=RBFSampler(), eta0=0.01, loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0, intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 learning_rate='pegasos', power_t=0.5, average=True,
                 warm_start=False, random_state=None, verbose=True,
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
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.average = average
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.fast_solver = fast_solver
        self.shuffle = shuffle

    def _init_params(self, X):
        super(BaseSGDEstimator, self)._init_params(X)
        n_components = self.transformer.n_components

        if not (self.warm_start and hasattr(self, 'coef_cache_')):
            self.coef_cache_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'intercept_cache_')):
            self.intecept_cache_ = np.zeros(1)

    def _valid_params(self):
        super(BaseSGDEstimator, self)._valid_params()

        if not isinstance(self.average, bool):
            raise ValueError("average is not bool.")

        if self.learning_rate == 'pegasos':
            if not (self.l1_ratio < 1):
                raise ValueError("1 - l1_ratio must be > 0 when "
                                 "learning_rate = 'pegasos'.")

            if not (self.alpha / self.C > 0):
                raise ValueError("alpha / C must be > 0 when "
                                 "learning_rate = 'pegasos'.")

            if not (self.intercept_decay / self.C > 0):
                raise ValueError("intercept_decay / C must be > 0 when "
                                 "learning_rate = 'pegasos'.")

        if not (0 <= self.power_t):
            raise ValueError("power_t < 0.")

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
        # valid hyper parameters and init parameters
        self._valid_params()
        self._init_params(X)

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        intercept_decay = self.intercept_decay / self.C
        random_state = check_random_state(self.random_state)
        is_sparse = sparse.issparse(X)
        learning_rate = self.LEARNING_RATE[self.learning_rate]
        it = _sgd_fast(self.coef_, self.intercept_, self.coef_cache_,
                       self.intecept_cache_, get_dataset(X, order='c'), X, y,
                       self.mean_, self.var_, loss, alpha, self.l1_ratio,
                       intercept_decay, self.eta0, learning_rate,
                       self.power_t, self.average, self.t_, self.max_iter,
                       self.tol, is_sparse, self.verbose, self.fit_intercept,
                       self.shuffle, random_state, self.transformer,
                       get_fast_random_feature(self.transformer))
        self.t_ += n_samples*(it+1)

        return self


class SGDClassifier(BaseSGDEstimator, LinearClassifierMixin):
    """SGD solver for linear classifier with random feature maps.

    Random feature mapping is computed just before computing prediction and
    gradient.
    minimize  \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object (default=RBFSampler())
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X).

    eta0 : double (default=0.01)
        Step-size parameter.

    loss : str (default="squared_hinge")
        Which loss function to use. Following losses can be used:

        - 'squared_hinge' (for classification)

        - 'hinge' (for classification)

        - 'logistic' (for classification)

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
        If true, the sgd solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool (default=True)
        Whether to fit intercept (bias term) or not.

    max_iter : int (default=100)
        Maximum number of iterations.

    tol : double (default=1e-6)
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the SGD solver stops learning.

    learning_rate : str (default='optimal')
        The method for learning rate decay.

        - 'constant': eta = eta0

        - 'pegasos': eta = 1.0 / (alpha * (1-l1_ratio) * t)

        - 'inv_scaling': eta = eta0 / pow(t, power_t)

        - 'optimal': eta = eta0 / pow(1 + alpha*(1-l1_ratio)*t, power_t)

    power_t : double (default=0.5)
        The parameter for learning_rate 'inv_scaling'.

    average : bool (default=True)
        Whether output averaged weight or not.
        If average=True, you should use learning_rate='optimal' and
        power_t=0.75[2].

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

    self.coef_cache_ : array, shape (n_components, )
        The latest learned coefficients of the linear model for average=True.

    self.intercept_ : array, shape (1, )
        The learned intercept (bias) of the linear model.

    self.intercept_cache_ : array, shape (1, )
        The latest learned intercept (bias) of the linear model
        for average=True.

    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    self.t_ : int
        The number of iteration.

    References
    ----------
    [1] Large-Scale Machine Learning with Stochastic Gradient Descent.
    Leon Bottou.
    In Proc. COMPSTAT'2010.
    (https://leon.bottou.org/publications/pdf/compstat-2010.pdf)

    [2] Stochastic Gradient Descent Tricks.
    Leon Bottou.
    Neural Networks, Tricks of the Trade, Reloaded, 430–445,
    Lecture Notes in Computer Science (LNCS 7700), Springer, 2012
    (https://link.springer.com/content/pdf/10.1007%2F978-3-642-35289-8_25.pdf)

    """

    LOSSES = {
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge(),
        'log': Logistic()
    }

    def __init__(self, transformer=RBFSampler(), eta0=0.01,
                 loss='squared_hinge', C=1.0, alpha=1.0, l1_ratio=0.,
                 intercept_decay=0.1, normalize=False, fit_intercept=True,
                 max_iter=100, tol=1e-6, learning_rate='optimal', power_t=0.75,
                 average=True, warm_start=False, random_state=None,
                 verbose=True, fast_solver=True, shuffle=True):
        super(SGDClassifier, self).__init__(
            transformer, eta0, loss, C, alpha, l1_ratio, intercept_decay,
            normalize, fit_intercept, max_iter, tol, learning_rate, power_t,
            average, warm_start, random_state, verbose, fast_solver, shuffle
        )


class SGDRegressor(BaseSGDEstimator, LinearRegressorMixin):
    """SGD solver for linear regression with random feature maps.

    Random feature mapping is computed just before computing prediction and
    gradient.
    minimize  \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object (default=RBFSampler())
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X).

    eta0 : double (default=0.01)
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

        - l1_ratio = 1 : Lasso.

        - Otherwise : Elastic Net.

    intercept_decay : double (default=0.1)
        Weight of the penalty term for intercept.

    normalize : bool (default=False)
        Whether normalize random features or not.
        If true, the sgd solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool (default=True)
        Whether to fit intercept (bias term) or not.

    max_iter : int (default=100)
        Maximum number of iterations.

    tol : double (default=1e-6)
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the SGD solver stops learning.

    learning_rate : str (default='optimal')
        The method for learning rate decay.

        - 'constant': eta = eta0

        - 'pegasos': eta = 1.0 / (alpha * (1-l1_ratio) * t)

        - 'inv_scaling': eta = eta0 / pow(t, power_t)

        - 'optimal': eta = eta0 / pow(1 + alpha*(1-l1_ratio)*t, power_t)

    power_t : double (default=0.5)
        The parameter for learning_rate 'inv_scaling'.

    average : bool (default=True)
        Whether output averaged weight or not.
        If average=True, you should use learning_rate='optimal' and
        power_t=0.75[2].

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
        is implemented in random_features_fast.pyx/pxd.

    shuffle : bool (default=True)
        Whether to shuffle data before each epoch or not.

    Attributes
    ----------
    self.coef_ : array, shape (n_components, )
        The learned coefficients of the linear model.

    self.coef_cache_ : array, shape (n_components, )
        The latest learned coefficients of the linear model for average=True.

    self.intercept_ : array, shape (1, )
        The learned intercept (bias) of the linear model.

    self.intercept_cache_ : array, shape (1, )
        The latest learned intercept (bias) of the linear model
        for average=True.

    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    self.t_ : int
        The number of iteration.

    References
    ----------
    [1] Large-Scale Machine Learning with Stochastic Gradient Descent.
    Leon Bottou.
    In Proc. COMPSTAT'2010.
    (https://leon.bottou.org/publications/pdf/compstat-2010.pdf)

    [2] Stochastic Gradient Descent Tricks.
    Leon Bottou.
    Neural Networks, Tricks of the Trade, Reloaded, 430–445, 
    Lecture Notes in Computer Science (LNCS 7700), Springer, 2012
    (https://link.springer.com/content/pdf/10.1007%2F978-3-642-35289-8_25.pdf)

    """
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, transformer=RBFSampler(), eta0=0.001, loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0., intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 learning_rate='optimal', power_t=0.75, average=True,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(SGDRegressor, self).__init__(
            transformer, eta0, loss, C, alpha, l1_ratio, intercept_decay,
            normalize, fit_intercept, max_iter, tol, learning_rate, power_t,
            average, warm_start, random_state, verbose, fast_solver, shuffle
        )
