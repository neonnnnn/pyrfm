# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs, sqrt
from .loss_fast cimport LossFunction
import numpy as np
cimport numpy as np


def _adagrad_fast(double[:] coef,
                  double[:] intercept,
                  X,
                  double[:] y,
                  double[:] acc_grad,
                  double[:] acc_grad_norm,
                  double[:] acc_grad_intercept,
                  double[:] acc_grad_norm_intercept,
                  double[:] mean,
                  double[:] var,
                  LossFunction loss,
                  double alpha,
                  double l1_ratio,
                  double eta,
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
    cdef double dloss, eta_t, viol, y_pred, denom, mean_new
    cdef double intercept_new, coef_new_j, lam1, lam2
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
    n_samples = X.shape[0]
    n_components = coef.shape[0]

    cdef int[:] indices = np.arange(n_samples, dtype=np.int32)
    cdef double[:] x = np.zeros((n_components, ), dtype=np.float64)
    it = 0

    random_state.shuffle(indices)
    if mean is not None and t == 1:
        i = indices[random_state.randint(n_samples-1)+1]

        if is_sparse:
            x = transformer.transform(X[i])[0]
        else:
            x = transformer.transform(np.atleast_2d(X[i]))[0]
        for j in range(n_components):
            mean[j] = x[j]

    for it in range(max_iter):
        viol = 0
        if it != 0:
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
            for j in range(n_components):
                y_pred += x[j] * coef[j]
            y_pred += intercept[0]

            # update parameters
            dloss = loss.dloss(y_pred, y[i])
            eta_t = eta*t
            if dloss != 0:
                for j in range(n_components):
                    acc_grad[j] += dloss * x[j]
                    acc_grad_norm[j] += (dloss*x[j])**2

            for j in range(n_components):
                denom = sqrt(acc_grad_norm[j]) + eps + lam2*eta_t
                if fabs(acc_grad[j])/t - lam1 < 0:
                    coef_new_j = 0
                else:
                    coef_new_j = -eta_t / denom
                    if acc_grad[j] > 0:
                        coef_new_j *= (acc_grad[j]/t - lam1)
                    else:
                        coef_new_j *= (acc_grad[j]/t + lam1)

                viol += fabs(coef[j] - coef_new_j)
                coef[j] = coef_new_j

            if fit_intercept:
                acc_grad_intercept[0] += dloss
                acc_grad_norm_intercept[0] += dloss*dloss
                denom = sqrt(acc_grad_norm_intercept[0]) + eps
                intercept_new = -eta_t*acc_grad_intercept[0] / t
                intercept_new /= denom
                viol += fabs(intercept_new - intercept[0])
                intercept[0] = intercept_new
            t += 1

        if verbose:
            print("Iteration {} Violation {}".format(it, viol))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it))
            break

    return it