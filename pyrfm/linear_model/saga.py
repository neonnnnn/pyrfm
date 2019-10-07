# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin
from sklearn.kernel_approximation import RBFSampler
from .saga_fast import _saga_fast
from ..dataset_fast import get_dataset
from ..random_feature.random_features_fast import get_fast_random_feature


class BaseSAGAEstimator(BaseLinear):
    LEARNING_RATE = {
        'constant': 0,
        'pegasos': 1,
        'inv_scaling': 2
    }
    """SAGA solver for linear models with random feature maps.
    Random feature mapping is computed just before computing prediction and
    gradient.
    minimize  \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object (default=RBFSampler())
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X).

    eta0 : double (default=1.0)
        Step-size parameter.

    loss : str (default="squared")
        Which loss function to use. Following losses can be used:
            'squared' (for regression)
            'squared_hinge' (for classification)
            'hinge' (for classification)
            'logistic' (for classification)

    C : double (default=1.0)
        Weight of the loss term.

    alpha : double (default=1.0)
        Weight of the penalty term.

    l1_ratio : double (default=0)
        Ratio of L1 regularizer.
        Weight of L1 regularizer is alpha * l1_ratio and that of L2 regularizer
        is 0.5 * alpha * (1-l1_ratio).
        If l1_ratio = 0 : Ridge.
        else If l1_ratio = 1 : Lasso.
        else : Elastic Net.

    normalize : bool (default=False)
        Whether normalize random features or not.
        If true, the SAGA solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool (default=True)
        Whether to fit intercept (bias term) or not.

    max_iter : int (default=100)
        Maximum number of iterations.

    tol : double (default=1e-6)
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the SAGA solver stops learning.

    learning_rate : str (default='pegasos')
        The method for learning rate decay. {'constant'|'pegasos'|'inv_scaling'}
        are supported now.
    
    power_t : double (default=0.5)
        The parameter for learning_rate 'inv_scaling'.
    
    is_saga : bool (default=True)
        Whether SAGA (True) or SAG (False).
    
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
        is in {RandomFourier|RandomMaclaurin|TensorSketch|RandomKernel}.

    shuffle : bool (default=True)
        Whether shuffle data before each epoch or not.

    Attributes
    ----------
    self.coef_ : array, shape (n_components, )
        The learned coefficients of the linear model.
        
    self.averaged_grad_coef_ : array, shape (n_components, )
        The averaged gradient of coefficients.
    
    self.intercept_ : array, shape (1, )
        The learned intercept (bias) of the linear model.

    self.averaged_grad_intercept_ : array, shape (1, )
        The averaged gradient of intercept.

    self.dloss : array, shape (n_samples, )
        The gradient of loss for each samples.    
                
    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    self.t_ : int
        The number of iteration.

    References
    ---------
    [1] SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly
    Convex Composite Objectives.
    Aaron Defazo, Francis Bach, and Simon Lacoste-Julien.
    In Proc. NIPS 2014.
    (https://arxiv.org/pdf/1407.0202.pdf)
    """
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
                 learning_rate='pegasos', power_t=0.5, is_saga=True,
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
        self.is_saga = is_saga
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.fast_solver = fast_solver
        self.shuffle = shuffle

    def _init_params(self, n_components):
        super(BaseSAGAEstimator, self)._init_params(n_components)

        if not (self.warm_start and hasattr(self, 'averaged_grad_coef')):
            self.averaged_grad_coef_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'averaged_grad_intercept_')):
            self.averaged_grad_intercept_ = np.zeros(1)

        if not (self.warm_start and hasattr(self, "dloss_")):
            self.dloss_ = np.zeros(n_components)

    def _valid_params(self):
        super(BaseSAGAEstimator, self)._valid_params()
        if not isinstance(self.is_saga, bool):
            raise ValueError("is_saga is not bool.")

        if self.learning_rate == 'pegasos':
            if not (self.l1_ratio > 0):
                raise ValueError("l1_ratio must be > 0 when "
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
        if not self.warm_start:
            self.transformer.fit(X)

        n_samples, n_features = X.shape
        n_components = self.transformer.n_components
        # valid hyper parameters and init parameters
        self._valid_params()
        self._init_params(n_components)

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        intercept_decay = self.intercept_decay / self.C
        random_state = check_random_state(self.random_state)
        is_sparse = sparse.issparse(X)
        learning_rate = self.LEARNING_RATE[self.learning_rate]

        it = _saga_fast(self.coef_, self.intercept_, self.averaged_grad_coef_,
                        self.averaged_grad_intercept_, self.dloss_,
                        get_dataset(X, order='c'), X, y, self.mean_, self.var_,
                        loss, alpha, self.l1_ratio, intercept_decay, self.eta0,
                        learning_rate, self.power_t, self.is_saga, self.t_,
                        self.max_iter, self.tol, is_sparse, self.verbose,
                        self.fit_intercept, self.shuffle, random_state,
                        self.transformer,
                        get_fast_random_feature(self.transformer))
        self.t_ += n_samples * (it + 1)

        return self


class SAGAClassifier(BaseSAGAEstimator, LinearClassifierMixin):
    LOSSES = {
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge(),
        'log': Logistic()
    }

    def __init__(self, transformer=RBFSampler(), eta0=1.0, loss='squared_hinge',
                 C=1.0, alpha=1.0, l1_ratio=0., intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 learning_rate='pegasos', power_t=0.5, is_saga=False,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(SAGAClassifier, self).__init__(
            transformer, eta0, loss, C, alpha, l1_ratio, intercept_decay,
            normalize, fit_intercept, max_iter, tol, learning_rate, power_t,
            is_saga, warm_start, random_state, verbose, fast_solver, shuffle
        )


class SAGARegressor(BaseSAGAEstimator, LinearRegressorMixin):
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, transformer=RBFSampler(), eta0=1.0, loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0., intercept_decay=0.1,
                 normalize=False, fit_intercept=True, max_iter=100, tol=1e-6,
                 learning_rate='pegasos', power_t=0.5, is_saga=False,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(SAGARegressor, self).__init__(
            transformer, eta0, loss, C, alpha, l1_ratio, intercept_decay,
            normalize, fit_intercept, max_iter, tol, learning_rate, power_t,
            is_saga, warm_start, random_state, verbose, fast_solver, shuffle
        )
