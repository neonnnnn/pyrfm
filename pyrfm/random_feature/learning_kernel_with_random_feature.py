# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.kernel_approximation import RBFSampler
from scipy.sparse import csc_matrix
from sklearn.utils.extmath import safe_sparse_dot
from scipy.optimize import linprog
import warnings
import abc
from .learning_kernel_with_random_feature_fast import proj_l1ball
from math import sqrt


class LearningKernelwithRandomFeature(BaseEstimator, TransformerMixin):
    """ Learnes importance weights for random features by maximizing the 
    kernel alignment.

    Parameters
    ----------
    transformer : sklearn transformer object (default=None)
        A random feature map object.
        If None, RBFSampler is used.

    divergence : str (default='chi2')
        Which f-divergence to use.
        
        - 'chi2': (p/q)^2 -1

        - 'kl': (p/q log(p/q))

        - 'tv': |p/q-1|/2

        - 'squared': 0.5*(p-q)^2

    max_iter : int (default=100)
        Maximum number of iterations.

    rho : double (default=1.)
        A upper bound of divergence.
        If rho=0, the importance weights will be 1/\sqrt{n_components}.

    alpha : double (default=None)
        A strenght hyperparameter for divergence.
        If not None, optimize the regularized objective (+ alpha * divergence).
        If None, optimize the constraied objective (divergence < rho).

    tol : double (default=1e-8)
        Tolerance of stopping criterion.

    warm_start : bool (default=False)
        Whether to active warm-start or not.

    max_iter_admm : int (default=10000)
        Maximum number of iterations in the ADMM optimization.
        This is used if divergence='tv' or 'reverse_kl'.
    
    mu : double (default=10)
        A parameter for the ADMM optimization.
        Larger mu updates the penalty parameter more frequently.
        This is used if divergence='tv' or 'reverse_kl'.

    tau_incr : double (default=2)
        A parameter for the ADMM optimization.
        The penalty parameter updated by multiplying tau_incr.
        This is used if divergence='tv' or 'reverse_kl'.

    tau_decr : double (default=2)
        A parameter for the ADMM optimization.
        The penalty parameter updated by dividing tau_decr.
        This is used if divergence='tv' or 'reverse_kl'.
        
    eps_abs : double (default=1e-4)
        A parameter for the ADMM optimization.
        It is used for stopping criterion.
        This is used if divergence='tv' or 'reverse_kl'.
     
    eps_rel : double (default=1e-4)
        A parameter for the ADMM optimization.
        It is used for stopping criterion.
        This is used if divergence='tv' or 'reverse_kl'.

    verbose : bool (default=True)
        Verbose mode or not.

    Attributes
    ----------
    importance_weights_ : array, shape (n_components, )
        The learned importance weights.
    
    divergence_ : Divergence instance
        The divergence instance for optimization.

    References
    ----------
    [1] Learning Kernels with Random Features.
    Aman Sinha and John Duchi.
    In NIPS 2016.
    (https://papers.nips.cc/paper/6180-learning-kernels-with-random-features.pdf)

    """
    def __init__(self, transformer=None, divergence='chi2', rho=1.,
                 alpha=None, max_iter=100, tol=1e-8, warm_start=False,
                 max_iter_admm=10000, mu=10, tau_incr=2, tau_decr=2,
                 eps_abs=1e-4, eps_rel=1e-4, verbose=True):
        self.transformer = transformer
        self.divergence = divergence
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha
        self.tol = tol
        self.warm_start = warm_start
        self.max_iter_admm = max_iter_admm
        self.verbose = verbose
        self.mu = mu
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr
        self.eps_abs = eps_abs
        self.eps_rel =  eps_rel

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y_kind = np.unique(y)
        if not (y_kind[0] == -1 and y_kind[1] == 1):
            raise ValueError("Each element in y must be -1 or 1.")

        if not (self.warm_start and hasattr(self, "divergence_")):
            self.divergence_ = self._get_divergence()
       
        if self.transformer is None:
            self.transformer = RBFSampler()
        if not (self.warm_start 
                and hasattr(self.transformer, "random_weights_")):
            self.transformer.fit(X)
        X_trans = self.transformer.transform(X)

        if not (self.warm_start and hasattr(self, "importance_weights_")):
            self.importance_weights_ = np.zeros(X_trans.shape[1])

        scale = np.sqrt(X_trans.shape[1])
        v = safe_sparse_dot(y, X_trans*scale, dense_output=True)**2

        self.divergence_.fit(self.importance_weights_, v)
        return self
    
    def transform(self, X):
        X_transform = self.transformer.transform(X)
        X_transform *= np.sqrt(X_transform.shape[1]) 
        return X_transform * np.sqrt(self.importance_weights_)

    def _get_divergence(self):
        if self.divergence == 'chi2':
            return Chi2(self.rho, self.alpha, self.tol, self.warm_start,
                        self.max_iter, self.verbose)
        elif self.divergence == 'kl':
            return KL(self.rho, self.alpha, self.tol, self.warm_start, 
                      self.max_iter, self.verbose)
        elif self.divergence == 'tv':
            return TV(self.rho, self.alpha, self.tol, self.warm_start, 
                      self.max_iter, self.verbose, self.max_iter_admm, self.mu,
                      self.tau_incr, self.tau_decr, self.eps_abs, self.eps_rel)
        elif self.divergence == 'chi2_origin':
            return Chi2Origin(self.rho, self.alpha, self.tol, self.warm_start, 
                              self.max_iter, self.verbose)
        elif self.divergence == 'squared':
            return Squared(self.rho, self.alpha, self.tol, self.warm_start, 
                           self.max_iter, self.verbose)
        elif self.divergence == 'reverse_kl':
            return ReverseKL(self.rho, self.alpha, self.tol, self.warm_start, 
                             self.max_iter, self.verbose, self.max_iter_admm,
                             self.mu, self.tau_incr, self.tau_decr,
                             self.eps_abs, self.eps_rel)
        else:
            raise ValueError("f={} is not supported now. Use {'chi2'|'kl'|"
                             "'tv'|'reverse_kl'|'squared'}."
                             .format(self.divergence))


class BaseDivergence(metaclass=abc.ABCMeta):
    def __init__(self, rho, alpha, tol, warm_start, max_iter, verbose):
        self.rho = rho
        self.alpha = alpha
        self.tol = tol
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, importance_weights_, v):
        if not (self.warm_start and hasattr(self, "lam_u_")):
            self.lam_u_ = np.infty
        if not (self.warm_start and hasattr(self, "lam_l_")):
            self.lam_l_ = 0.
        if not (self.warm_start and hasattr(self, "lam_s_")):
            self.lam_s_ = 1.
        
        uniform = np.ones(len(v)) / len(v)
        if self.alpha is None:
            while self.lam_u_ == np.infty:
                self._fit(importance_weights_, v, self.lam_s_)
                div = self.eval(importance_weights_, uniform) 
                if self.verbose:
                    objective = np.dot(importance_weights_, v)
                    print("Objective: {} Divergence: {} lambda_s: {}"
                          .format(objective, div, self.lam_s_))

                if div < self.rho:
                    self.lam_u_ = self.lam_s_
                else:
                    self.lam_s_ = 2 * self.lam_s_
                
            for i in range(self.max_iter):
                lam = (self.lam_u_ + self.lam_l_) / 2.
                self._fit(importance_weights_, v, lam)
                div = self.eval(importance_weights_, uniform) 
                if self.verbose:
                    objective = np.dot(importance_weights_, v)
                    print("Iteration: {}  Objective: {} Divergence: {}"
                          .format(i+1, objective, div)
                    )
    
                if div < self.rho:
                    self.lam_u_ = lam
                else:
                    self.lam_l_ = lam
                
                if (self.lam_u_ - self.lam_l_) < self.tol * self.lam_s_:
                    if div < np.inf:
                        print("Converged.")
                        break
        else:
            if self.alpha <= 0:
                raise ValueError("alpha must be bigger than 0.")
            self._fit(self.importance_weights_, v, self.alpha)
            if self.verbose:
                div = self.eval(self.importance_weights_, uniform) 
                objective = np.dot(self.importance_weights_, v)
                print("Objective: {} Divergence: {}".format(objective, div))
        return self

    @abc.abstractmethod
    def eval(self, p, q):
        pass
    
    def __call__(self, p, q):
        return self.divergence(p, q)


class BaseADMMDivergence(BaseDivergence):
    """Base class for 
    minimizing -p^T v + lam*D(z-1/n, 1/n) + y^T(p-z-1/n) + rho*||p-q-1/n||_2^2/2
    by ADMM.

    """
    def __init__(self, rho, alpha, tol, warm_start, max_iter, verbose,
                 max_iter_admm, mu, tau_incr, tau_decr, eps_abs, eps_rel):
        super(BaseADMMDivergence, self).__init__(rho, alpha, tol, warm_start, max_iter, verbose)
        self.max_iter_admm = max_iter_admm
        self.mu = mu
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr
        self.eps_abs = eps_abs
        self.eps_rel =  eps_rel

    def _fit(self, p, v, lam):
        # optimize by ADMM
        multipler = 1.0
        n = len(v)
        z = np.ones(n)/n # augmented parameter
        y = np.zeros(n) # Lagrangian
        for _ in range(self.max_iter_admm):
            # update x
            p[:] = proj_l1ball(v - y + multipler*z, multipler) / multipler
            # update z (augmented parameter, x - (z) = 0)
            z_new = self._optimize_z(p, n, y, multipler, lam)
            norm_residual_p = sqrt(np.sum((p - z_new)**2))
            norm_residual_d = sqrt(np.sum(multipler * (z - z_new))**2)
            z = z_new
            # update Lagrangian
            y += multipler * (p - z)
            # stopping criterion
            norm_p = sqrt(np.dot(p, p))
            norm_z = sqrt(np.dot(z, z))
            norm_y = sqrt(np.dot(y, y))
            upper_primal = sqrt(n) * self.eps_abs
            upper_primal += self.eps_rel * max(norm_p, norm_z)
            upper_dual = sqrt(n) * self.eps_abs
            upper_dual += self.eps_rel * norm_y
            if norm_residual_p < upper_primal and norm_residual_d < upper_dual:
                break
            if norm_residual_p > norm_residual_d * self.mu:
                multipler *= self.tau_incr
            elif norm_residual_d > norm_residual_p * self.mu:
                multipler /= self.tau_decr

    def eval(self, p, q):
        return 0.5*np.sum(np.abs(p-q))


class Chi2(BaseDivergence):
    def _fit(self, p, v, lam):
        scale = 2*lam*len(v)
        p[:] = proj_l1ball(v, scale) / scale
        return self

    def eval(self, p, q):
        return np.dot((p/q)**2 - 1, q)


class Chi2Origin(BaseDivergence):
    def _fit(self, p, v, lam):
        scale = lam*len(v)
        p[:] = proj_l1ball(v/scale, 1) 
        return self

    def eval(self, p, q):
        return np.dot((p/q)**2 - 1, q)


class KL(BaseDivergence):
    def _fit(self, p, v, lam):
        p[:] = np.exp((v - np.max(v))/lam)
        p /= np.sum(p)
        return self

    def eval(self, p, q):
        t = p/q
        return np.dot(np.log(t[t!=0]), p[t!=0])


class TV(BaseADMMDivergence):
    """Total variation divergence.

    """
    def _optimize_z(self, p, n, y, multipler, lam):
        # apply soft-threasholding
        z_new = (multipler * (p-1./n) + y) / multipler
        z_new -= np.sign(z_new) * (1.0*lam / (2*multipler))
        z_new[np.abs(z_new) <= (1.0*lam / (2*multipler))] = 0
        return z_new + 1./n

    def eval(self, p, q):
        return 0.5*np.sum(np.abs(p-q))


class ReverseKL(BaseADMMDivergence):
    def _optimize_z(self, p, n, y, multipler, lam):
        a = multipler * n
        b = n * (y + multipler * p)
        return (b + np.sqrt(b*b + 4*a*lam)) / (2*a)

    def eval(self, p, q):
        return np.dot(q, np.log(q) - np.log(p+1e-8))


class Squared(BaseDivergence):
    def _fit(self, p, v, lam):
        scale = lam / len(v)
        offset = lam / (len(v)**2)
        p[:] = proj_l1ball(v+offset, scale) / scale
        return self

    def eval(self, p, q):
        return np.dot(p-q, p-q)*0.5
