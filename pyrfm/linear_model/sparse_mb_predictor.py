import numpy as np
from scipy import sparse


from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from ..dataset_fast import get_dataset

from .loss_fast import Squared, SquaredHinge, Logistic, Hinge
from ..random_feature.maji_berg import SparseMB
from .cd_primal_sparse_mb import _cd_primal
from .base import BaseLinear, LinearClassifierMixin, LinearRegressorMixin


class BaseSparseMBEstimator(BaseLinear):
    """Linear model with feature map approximating the intersection (min)
    kernel by sparse explicit feature map, which was proposed by S.Maji
    and A.C.Berg. SparseMB does not approximate min kernel only itself.
    Linear classifier with SparseMB approximates linear classifier with MB.
    For more detail, see [1].

    Parameters
    ----------
    n_components : int (default=1000)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    loss : str (default="squared")
        Which loss function to use. Following losses can be used:
            'squared' (for regression)
            'squared_hinge' (for classification)
            'logistic' (for classification)

    C : double (default=1.0)
        Weight of loss term.

    alpha : double (default=1.0)
        Weight of the penalty term.

    fit_intercept : bool (default=True)
        Whether to fit intercept (bias term) or not.

    max_iter : int (default=100)
        Maximum number of iterations.

    tol : double (default=1e-6)
        Tolerance of stopping criterion.
        If sum of absolute val of update in one epoch is lower than tol,
        the AdaGrad solver stops learning.

    eps : double (default=1e-2)
        A small double to ensure objective function convex.

    warm_start : bool (default=False)
        Whether to activate warm-start or not.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool (default=True)
        Verbose mode or not.

    shuffle : boole (default=True)
        Whether shuffle the order of parameters for optimization or not.

    Attributes
    ----------
    self.transformer_ : scikit-learn TransformMixin object.
        The learned transformer for random feature maps.

    self.coef_ : array, shape (n_components, )
        The learned coefficients of the linear model.

    self.intercept_ : array, shape (1, )
        The learned intercept (bias) of the linear model.


    References
    ----------
    [1] Max-Margin Additive Classifiers for Detection
    Subhransu Maji, Alexander C. Berg.
    In ICCV 2009.
    (http://acberg.com/papers/mb09iccv.pdf)
    """
    LOSSES = {
        'squared': Squared(),
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, n_components=1000, loss='squared_hinge', solver='cd',
                 C=1.0, alpha=1.0, fit_intercept=True, max_iter=100, tol=1e-6,
                 eps=1e-2, warm_start=False, random_state=None, verbose=True,
                 shuffle=True):
        self.n_components = n_components
        self.loss = loss
        # TODO Implement Group Lasso
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
        self.shuffle = shuffle

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
        X, y = self._check_X_y(X, y)
        if not (self.warm_start and hasattr(self, 'transformer_')):
            self.transformer_ = SparseMB(n_components=self.n_components)
            self.transformer_.fit(X)
        X_trans = self.transformer_.transform(X)
        if not (self.warm_start and hasattr(self, 'coef_')):
            self.coef_ = np.zeros(self.n_components)

        if not (self.warm_start and hasattr(self, 'intercept_')):
            self.intercept_ = 0.

        if self.loss not in self.LOSSES:
            raise ValueError("loss {} is not supported.".format(self.loss))

        loss = self.LOSSES[self.loss]
        alpha = self.alpha / self.C
        # make tridiagonal matrix
        H = sparse.diags([-1, 2+self.eps, -1], [-1, 0, 1],
                         shape=(self.n_components, self.n_components)).tocsr()
        H[0, 0] = 1+self.eps
        H[0, 1] = -1
        H[self.n_components-1, self.n_components-1] = 1+self.eps
        H[self.n_components-1, self.n_components-2] = -1
        y_pred = self._predict(X)
        X_col_norms = row_norms(X_trans.T, squared=True)
        X_trans_dataset = get_dataset(X_trans, 'fortran')
        H_dataset = get_dataset(H, 'c')
        random_state = check_random_state(self.random_state)
        _cd_primal(self.coef_, self.intercept_, X_trans_dataset, y,
                   X_col_norms, y_pred, H_dataset, alpha, loss,
                   self.max_iter, self.tol, self.fit_intercept,
                   random_state, self.verbose, self.shuffle)


class SparseMBClassifier(BaseSparseMBEstimator, LinearClassifierMixin):
    LOSSES = {
        'squared_hinge': SquaredHinge(),
        'logistic': Logistic(),
        'hinge': Hinge()
    }

    def __init__(self, n_components=1000, loss='squared_hinge',
                 solver='cd', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-4, warm_start=False,
                 random_state=None, verbose=True, shuffle=True):
        super(SparseMBClassifier, self).__init__(
            n_components, loss, solver, C, alpha, fit_intercept,
            max_iter, tol, eps, warm_start, random_state, verbose, shuffle
        )


class SparseMBRegressor(BaseSparseMBEstimator, LinearRegressorMixin):
    LOSSES = {
        'squared': Squared(),
    }

    def __init__(self, n_components=1000, loss='squared',
                 solver='cd', C=1.0, alpha=1.0, fit_intercept=True,
                 max_iter=100, tol=1e-6, eps=1e-4, warm_start=False,
                 random_state=None, verbose=True, shuffle=True):
        super(SparseMBRegressor, self).__init__(
            n_components, loss, solver, C, alpha, fit_intercept,
            max_iter, tol, eps, warm_start, random_state, verbose, shuffle
        )
