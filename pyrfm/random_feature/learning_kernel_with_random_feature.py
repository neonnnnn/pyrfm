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
from .learning_kernel_with_random_feature_fast import (optimize_chi2,
                                                       optimize_kl,
                                                       optimize_tv)


class LearningKernelwithRandomFeature(BaseEstimator, TransformerMixin):
    """ Learnes importance weights for random features by maximizing the 
    kernel alignment.

    Parameters
    ----------
    transformer : sklearn transformer object (default=None)
        A random feature map object.
        If None, RBFSampler is used.

    divergence : str (default='chi2')
        Which divergence to use.
        
        - 'chi2': (p/q)^2 -1

        - 'kl': (p/q log(p/q))

        - 'tv': |p/q-1|/2
    
    max_iter : int (default=100)
        Maximum number of iterations.

    rho : double (default=1.)
        A upper bound of divergence.
        If rho=0, the importance weights will be 1/\sqrt{n_components}.

    alpha : double (default=None)
        A strenght hyperparameter for divergence.
        If not None, optimize the regularized objective (+ alpha * divergence) 
        at once.
        If None, optimize the constraied objective (divergence < rho).

    tol : double (default=1e-6)
        Tolerance of stopping criterion.

    warm_start : bool (default=False)
        Whether to active warm-start or not.

    verbose : bool (default=True)
        Verbose mode or not.

    Attributes
    ----------
    importance_weights_ : array, shape (n_components, )
        The learned importance weights.

    lam_u_ : double
        A Lagrangian coef.
    
    lam_l_ : double
        A Lagrangian coef.
    
    lam_s_ : double
        A Lagrangian coef.

    References
    ----------
    [1] Learning Kernels with Random Features.
    Aman Sinha and John Duchi.
    In NIPS 2016.
    (https://papers.nips.cc/paper/6180-learning-kernels-with-random-features.pdf)

    """
    def __init__(self, transformer=None, divergence='chi2', rho=1.,
                 alpha=None, max_iter=100, tol=1e-6, warm_start=False,
                 verbose=True):
        self.transformer = transformer
        self.divergence = divergence
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y_kind = np.unique(y)
        if not (y_kind[0] == -1 and y_kind[1] == 1):
            raise ValueError("Each element in y must be -1 or 1.")
        
        if self.transformer is None:
            self.transformer = RBFSampler()
        if not (self.warm_start and hasattr(self, "lam_u_")):
            self.lam_u_ = np.infty
        if not (self.warm_start and hasattr(self, "lam_l_")):
            self.lam_l_ = 0.
        if not (self.warm_start and hasattr(self, "lam_s_")):
            self.lam_s_ = 1.

        if not (self.warm_start 
                and hasattr(self.transformer, "random_weights_")):
            self.transformer.fit(X)

        X_trans = self.transformer.transform(X)
        if not (self.warm_start and hasattr(self, "importance_weights_")):
            self.importance_weights_ = np.zeros(X_trans.shape[1])

        scale = np.sqrt(X_trans.shape[1])
        v = safe_sparse_dot(y, X_trans*scale, dense_output=True)**2
        divergence = self._get_divergence()
        uniform = np.ones(X_trans.shape[1]) / X_trans.shape[1]

        if self.alpha is None:
            while self.lam_u_ == np.infty:
                divergence.fit(self.importance_weights_, v, self.lam_s_)
                div = divergence(self.importance_weights_, uniform) 
                if div < self.rho:
                    self.lam_u_ = self.lam_s_
                else:
                    self.lam_s_ = 2 * self.lam_s_
                
                if self.verbose:
                    objective = np.dot(self.importance_weights_, v)
                    print("Objective: {} Divergence: {} lambda_s: {}"
                          .format(objective, div, self.lam_s_))

            for i in range(self.max_iter):
                lam = (self.lam_u_ + self.lam_l_) / 2.
                divergence.fit(self.importance_weights_, v, lam)
                div = divergence(self.importance_weights_, uniform) 
                if self.verbose:
                    objective = np.dot(self.importance_weights_, v)
                    print("Iteration: {}  Objective: {} Divergence: {}"
                          .format(i+1, objective, div)
                    )
    
                if div < self.rho:
                    self.lam_u_ = lam
                else:
                    self.lam_l_ = lam
                
                if (self.lam_u_ - self.lam_l_) < self.tol * self.lam_s_:
                    print("Converged.")
                    break
        else:
            if self.alpha <= 0:
                raise ValueError("alpha must be bigger than 0.")
            divergence.fit(self.importance_weights_, v, self.alpha)
            if self.verbose:
                div = divergence(self.importance_weights_, uniform) 
                objective = np.dot(self.importance_weights_, v)
                print("Objective: {} Divergence: {}".format(objective, div))

        return self
    
    def transform(self, X):
        X_transform = self.transformer.transform(X)
        X_transform *= np.sqrt(X_transform.shape[1]) 
        return X_transform * np.sqrt(self.importance_weights_)

    def _get_divergence(self):
        if self.divergence == 'chi2':
            return Chi2()
        elif self.divergence == 'kl':
            return KL()
        elif self.divergence == 'tv':
            return TV()
        else:
            raise ValueError("f={} is not supported now."
                             " Use {'chi2'|'kl'|'tv'}.")


class BaseDivergence(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, p, v, lam):
        pass

    @abc.abstractmethod
    def divergence(self, p, q):
        pass
    
    def __call__(self, p, q):
        return self.divergence(p, q)


class Chi2(BaseDivergence):
    def fit(self, p, v, lam):
        optimize_chi2(p, v, lam, len(p))
        return self

    def divergence(self, p, q):
        return np.dot((p/q)**2 - 1, q)


class KL(BaseDivergence):
    def __init__(self):
        pass

    def fit(self, p, v, lam):
        p[:] = np.exp((v - np.max(v))/lam)
        p /= np.sum(p)
        return self

    def divergence(self, p, q):
        t = p/q
        return np.dot(np.log(t[t!=0]), p[t!=0])


class TV(BaseDivergence):
    def __init__(self):
        pass

    def fit(self, p, v, lam):
        n = len(v)
        c = np.append(-v, 0.5*np.ones(n)*lam)
        A_eq = np.zeros((1, n*2))
        A_eq[0, :n] = 1
        b_eq = 1
        if not hasattr(self, "A_ub_"):
            A_ub1 = np.hstack((np.eye(n)*n, -np.eye(n)*n))
            A_ub2 = np.hstack((-np.eye(n)*n, -np.eye(n)*n))
            self.A_ub_ = np.vstack((A_ub1, A_ub2))
            self.b_ub_ = np.append(np.zeros(n), -np.zeros(n)) / n
        result = linprog(c, A_ub=self.A_ub_, b_ub=self.b_ub_,
                         A_eq=A_eq, b_eq=b_eq)
        p[:] = result['x'][:n]
        return self

    def divergence(self, p, q):
        return np.dot(0.5*np.abs(p-q), q)
