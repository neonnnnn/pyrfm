# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs, sqrt
from .loss_fast cimport LossFunction
import numpy as np
cimport numpy as np



cdef void _sgd_initialization(double[:] coef,
                              double[:] dual_coef,
                              double[:] intercept,
                              X,
                              double[:] y,
                              double[:] mean,
                              double[:] var,
                              LossFunction loss,
                              double alpha,
                              double l1_ratio,
                              unsigned int t,
                              double tol,
                              double eps,
                              bint is_sparse,
                              bint fit_intercept,
                              transformer,
                              double[:] x,
                              int[:] indices,
                              ):
    cdef Py_ssize_t i, n_samples, n_components, j
    cdef double dloss, viol, y_pred, coef_old, mean_new, update
    cdef double lam1, lam2, norm
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
    n_samples = X.shape[0]
    n_components = coef.shape[0]
    # modified SGD
    for i in indices:
        if is_sparse:
            x = transformer.transform(X[i])[0]
        else:
            x = transformer.transform(np.atleast_2d(X[i]))[0]

        # if normalize
        if mean is not None:
            for j in range(n_components):
                mean_new = mean[j] + (x[j] - mean[j]) / (t+1)
                var[j] = var[j] * (1-1./t)
                var[j] += (x[j] - mean[j])*(x[j] - mean_new) / t
                mean[j] = mean_new
                x[j] = (x[j] - mean[j]) / (eps + sqrt(var[j]))

        y_pred = 0
        norm = 0
        for j in range(n_components):
            y_pred += x[j] * coef[j]
            norm += x[j]**2

        y_pred += intercept[0]
        if fit_intercept:
            norm += 1

        # update dual_coef
        update = loss.sdca_update(dual_coef[i], y[i], y_pred,
                                  norm / (lam2*t))
        dual_coef[i] += update

        # update primal coef
        for j in range(n_components):
            coef_old = coef[j]
            coef[j] *= lam2*(t-1)
            coef[j] += update * x[j]
            coef[j] /= lam2*t
            # proximal
            if coef[j] > lam1:
                coef[j] -= lam1
            elif coef[j] < -lam1:
                coef[j] += lam1
            else:
                coef[j] = 0

        if fit_intercept:
            intercept[0] *= lam2*(t-1)
            intercept[0] += update
            intercept[0] /= lam2*t
        t += 1


def _sdca_fast(double[:] coef,
               double[:] dual_coef,
               double[:] intercept,
               X,
               double[:] y,
               double[:] mean,
               double[:] var,
               LossFunction loss,
               double alpha,
               double l1_ratio,
               unsigned int t,
               unsigned int max_iter,
               double tol,
               double eps,
               bint is_sparse,
               bint verbose,
               bint fit_intercept,
               random_state,
               transformer,
               ):
    cdef Py_ssize_t it, i, n_samples, n_components, j
    cdef double dloss, viol, y_pred, coef_old, mean_new, update
    cdef double lam1, lam2, norm
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
    n_samples = X.shape[0]
    n_components = coef.shape[0]

    cdef int[:] indices = np.arange(n_samples, dtype=np.int32)
    cdef double[:] x = np.zeros((n_components, ), dtype=np.float64)
    it = 0

    if mean is not None and t == 1:
        i = random_state.randint(n_samples)

        if is_sparse:
            x = transformer.transform(X[i])[0]
        else:
            x = transformer.transform(np.atleast_2d(X[i]))[0]
        for j in range(n_components):
            mean[j] = x[j]

    # initialize by SGD
    if t == 1:
        random_state.shuffle(indices)
        _sgd_initialization(coef, dual_coef, intercept, X, y, mean, var, loss,
                            alpha, l1_ratio, t, tol, eps, is_sparse,
                            fit_intercept, transformer, x, indices)
    # start epoch
    for it in range(max_iter):
        viol = 0
        random_state.shuffle(indices)
        for i in indices:
            if is_sparse:
                x = transformer.transform(X[i])[0]
            else:
                x = transformer.transform(np.atleast_2d(X[i]))[0]

            # if normalize
            if mean is not None:
                for j in range(n_components):
                    mean_new = mean[j] + (x[j] - mean[j]) / (t+1)
                    var[j] = var[j] * (1-1./t)
                    var[j] += (x[j] - mean[j])*(x[j] - mean_new) / t
                    mean[j] = mean_new
                    x[j] = (x[j] - mean[j]) / (eps + sqrt(var[j]))

            y_pred = 0
            norm = 0
            for j in range(n_components):
                y_pred += x[j] * coef[j]
                norm += x[j]**2

            y_pred += intercept[0]
            if fit_intercept:
                norm += 1

            # update dual_coef
            update = loss.sdca_update(dual_coef[i], y[i], y_pred,
                                      norm / (lam2*n_samples))
            dual_coef[i] += update

            # update primal coef
            for j in range(n_components):
                coef_old = coef[j]
                coef[j] += update * x[j] / (n_samples*lam2)
                # proximal
                if coef[j] > lam1:
                    coef[j] -= lam1
                elif coef[j] < -lam1:
                    coef[j] += lam1
                else:
                    coef[j] = 0
                viol += fabs(coef_old-coef[j])
            if fit_intercept:
                intercept[0] += update / (n_samples*lam2)
            t += 1

        if verbose:
            print("Iteration {} Violation {}".format(it, viol))
        # TODO: use duality gap for stopping criterion
        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it))
            break

    return it