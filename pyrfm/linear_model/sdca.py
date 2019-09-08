import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin
from sklearn.kernel_approximation import RBFSampler
from .sdca_fast import _sdca_fast
from lightning.impl.dataset_fast import get_dataset
from ..random_feature.random_features_fast import get_fast_random_feature


class BaseSDCAEstimator(BaseLinear):
    """Stochastic dual coordinate ascent solver for linear models with
    random feature maps.
    Random feature mapping is computed just before computing prediction and
    gradient.
    minimize  \sum_{i=1}^{n} loss(x_i, y_i) + alpha/C*reg

    Parameters
    ----------
    transformer : scikit-learn Transformer object, default=RBFSampler()
        A scikit-learn TransformerMixin object.
        transformer must have (1) n_components attribute, (2) fit(X, y),
        and (3) transform(X) methods.

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
        If true, the SDCA solver computes running mean and variance
        at learning, and uses them for inference.

    fit_intercept : bool, default=True
        Whether to fit intercept (bias term) or not.

    max_iter : int
        Maximum number of iterations.

    tol : double
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the SDCA solver stops learning.

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

    shuffle : bool, default=True
        Whether shuffle data before each epoch or not.
        If shuffle=False, the SDCA becomes (Cyclic) DCA.

    Attributes
    ----------
    self.coef_ : array, shape (n_components, )
        The learned coefficients of the linear model.

    self.dual_coef_ : array, shape (n_samples, )

    self.intercept_ : array, shape (1, )
        The learned intercept (bias) of the linear model.

    self.mean_, self.var_ : array or None, shape (n_components, )
        The running mean and variances of random feature vectors.
        They are used if normalize=True (they are None if False).

    self.t_ : int
        The number of iteration.

    References
    ---------
    [1] Stochastic Dual Coordinate Ascent Methods for Regularized Loss
    Minimization.
    Shai Shalev-Schwartz and Tong Zhang.
    JMLR 2013 (vol 14), pp. 567-599.
    """

    LOSSES = {
        'squared': Squared(),
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }
    stochastic = True

    def __init__(self, transformer=RBFSampler(), loss='squared_hinge',
                 C=1.0, alpha=1.0, l1_ratio=0, normalize=False,
                 fit_intercept=True, max_iter=100, tol=1e-6,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        self.transformer = transformer
        self.transformer_ = transformer
        self.loss = loss
        self.C = C
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.fast_solver = fast_solver
        self.shuffle = shuffle

    def fit(self, X, y):
        X, y = self._check_X_y(X, y, accept_sparse=['csr'])
        if not self.warm_start:
            self.transformer.fit(X)

        n_samples, n_features = X.shape
        if not (hasattr(self.transformer, 'n_components_actual_')):
            n_components = self.transformer.n_components
        else:
            n_components = self.transformer.n_components_actual_
        # init primal parameters, mean/var vectors and t_
        self._init_params(n_components)

        self.dual_coef_ = np.zeros(n_samples)

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        random_state = check_random_state(self.random_state)

        is_sparse = sparse.issparse(X)
        it = _sdca_fast(self.coef_, self.dual_coef_, self.intercept_,
                        get_dataset(X, order='c'), X, y,
                        self.mean_, self.var_, loss, alpha,
                        self.l1_ratio, self.t_, self.max_iter, self.tol,
                        is_sparse, self.verbose, self.fit_intercept,
                        self.shuffle, random_state, self.transformer,
                        get_fast_random_feature(self.transformer))
        self.t_ += n_samples*(it+1)

        return self


class SDCAClassifier(BaseSDCAEstimator, LinearClassifierMixin):
    LOSSES = {
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge(),
        'log': Logistic()
    }

    def __init__(self, transformer=RBFSampler(), loss='squared_hinge',
                 C=1.0, alpha=1.0, l1_ratio=0., normalize=False,
                 fit_intercept=True, max_iter=100, tol=1e-6,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(SDCAClassifier, self).__init__(
            transformer, loss, C, alpha, l1_ratio, normalize,
            fit_intercept, max_iter, tol, warm_start, random_state, verbose,
            fast_solver, shuffle
        )


class SDCARegressor(BaseSDCAEstimator, LinearRegressorMixin):
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, transformer=RBFSampler(), loss='squared',
                 C=1.0, alpha=1.0, l1_ratio=0., normalize=False,
                 fit_intercept=True, max_iter=100, tol=1e-6,
                 warm_start=False, random_state=None, verbose=True,
                 fast_solver=True, shuffle=True):
        super(SDCARegressor, self).__init__(
            transformer, loss, C, alpha, l1_ratio, normalize,
            fit_intercept, max_iter, tol, warm_start, random_state, verbose,
            fast_solver, shuffle
        )
