# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from libc.math cimport fabs, sqrt
from .loss_fast cimport LossFunction
import numpy as np
cimport numpy as np
from ..dataset_fast cimport RowDataset
from cython.view cimport array
from .utils cimport transform, normalize
from ..random_feature.random_features_fast cimport BaseCRandomFeature


cdef inline double proximal(double coef,
                            double lam):
    if coef > lam:
        return coef - lam
    elif coef < -lam:
        return coef + lam
    else:
        return 0.


cdef inline double _pred(double[:] z,
                         double[:] coef,
                         double intercept,
                         double lam1,
                         double lam2,
                         double* acc_loss,
                         Py_ssize_t n_components):
    cdef double y_pred = 0
    cdef Py_ssize_t j
    for j in range(n_components):
        y_pred += z[j] * coef[j]
        acc_loss[0] += 0.5*lam2*coef[j]**2 + lam1*fabs(coef[j])
    return y_pred + intercept


cdef double adam_epoch(double[:] coef,
                       double[:] intercept,
                       RowDataset X,
                       X_array,
                       double[:] y,
                       double[:] mean_grad,
                       double[:] var_grad,
                       double[:] mean_grad_intercept,
                       double[:] var_grad_intercept,
                       double[:] mean,
                       double[:] var,
                       LossFunction loss,
                       double lam1,
                       double lam2,
                       double intercept_decay,
                       double eta,
                       double beta1,
                       double beta2,
                       unsigned int* t,
                       double eps,
                       bint is_sparse,
                       bint fit_intercept,
                       bint shuffle,
                       random_state,
                       double* acc_loss,
                       transformer,
                       BaseCRandomFeature transformer_fast,
                       np.ndarray[int, ndim=1] indices_samples,
                       double[:] z):

    cdef Py_ssize_t i, ii, j
    cdef int n_samples, n_components
    cdef double dloss, viol, y_pred, intercept_new, coef_new_j, denom, eta_t
    cdef double grad, m_hat_t, v_hat_t
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    viol = 0
    if shuffle:
        random_state.shuffle(indices_samples)

    if mean is not None and t[0] == 1:
        i = indices_samples[0]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)

        for j in range(n_components):
            mean[j] = z[j]

    for ii in range(n_samples):
        i = indices_samples[ii]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse,
                  transformer, transformer_fast)

        # if normalize
        if mean is not None:
            normalize(z, mean, var, t[0], n_components)

        y_pred = _pred(z, coef, intercept[0], lam1, lam2, &acc_loss[0],
                       n_components)
        acc_loss[0] += loss.loss(y_pred, y[i])

        # update parameters
        dloss = loss.dloss(y_pred, y[i])
        eta_t = eta * sqrt(1-beta2**t[0]) / (1-beta1**t[0])
        for j in range(n_components):
            grad = dloss*z[j] + lam2*coef[j]
            mean_grad[j] = beta1*mean_grad[j] + (1-beta1)*grad
            var_grad[j] = beta2*var_grad[j] + (1-beta2)*grad**2
            denom = sqrt(var_grad[j]) + eps
            coef_new_j = coef[j] - eta_t * mean_grad[j] / denom
            viol += fabs(coef[j] - coef_new_j)
            coef[j] = coef_new_j
            coef[j] = proximal(coef_new_j, eta*lam1/sqrt(t[0]))

        if fit_intercept:
            grad = dloss + intercept_decay*intercept[0]
            mean_grad_intercept[0] *= beta1
            mean_grad_intercept[0] += (1-beta1)*grad
            var_grad_intercept[0] *= beta2
            var_grad_intercept[0] += (1-beta2)*grad*grad

            m_hat_t = mean_grad_intercept[0] / (1-beta1**t[0])
            v_hat_t = var_grad_intercept[0] / (1-beta2**t[0])
            denom = sqrt(v_hat_t) + eps
            intercept_new = eta*mean_grad_intercept[0]/denom
            intercept_new = intercept[0] - intercept_new
            viol += fabs(intercept_new - intercept[0])
            intercept[0] = intercept_new

        t[0] += 1
    acc_loss[0] /= n_samples
    return viol


def _adam_fast(double[:] coef,
               double[:] intercept,
               RowDataset X,
               X_array,
               double[:] y,
               double[:] mean_grad,
               double[:] var_grad,
               double[:] mean_grad_intercept,
               double[:] var_grad_intercept,
               double[:] mean,
               double[:] var,
               LossFunction loss,
               double alpha,
               double l1_ratio,
               double intercept_decay,
               double eta,
               double beta1,
               double beta2,
               unsigned int t,
               unsigned int max_iter,
               double tol,
               double eps,
               bint is_sparse,
               bint verbose,
               bint fit_intercept,
               bint shuffle,
               random_state,
               transformer,
               BaseCRandomFeature transformer_fast):

    cdef Py_ssize_t it, n_samples, n_components, j
    cdef double viol, lam1, lam2, acc_loss
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    cdef np.ndarray[int, ndim=1] indices_samples = np.arange(n_samples,
                                                             dtype=np.int32)
    cdef double[:] z = array((n_components, ), sizeof(double), format='d')
    for j in range(n_components):
        z[j] = 0

    it = 0
    for it in range(max_iter):
        acc_loss = 0
        viol = adam_epoch(coef, intercept, X, X_array, y, mean_grad,
                          var_grad, mean_grad_intercept,
                          var_grad_intercept, mean, var, loss, lam1,
                          lam2, intercept_decay, eta, beta1, beta2, &t, eps,
                          is_sparse, fit_intercept, shuffle, random_state,
                          &acc_loss, transformer, transformer_fast,
                          indices_samples, z)
        if verbose:
            print("Iteration {} Violation {} Loss {}".format(it+1, viol,
                                                             acc_loss))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it+1))
            break

    return it