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
from .utils cimport transform, normalize
from cython.view cimport array
from ..random_feature.random_features_fast cimport BaseCRandomFeature


cdef double adagrad_epoch(double[:] coef,
                          double[:] intercept,
                          RowDataset X,
                          X_array,
                          double[:] y,
                          double[:] acc_grad,
                          double[:] acc_grad_norm,
                          double[:] acc_grad_intercept,
                          double[:] acc_grad_norm_intercept,
                          double[:] mean,
                          double[:] var,
                          LossFunction loss,
                          double lam1,
                          double lam2,
                          double eta,
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
    cdef double dloss, eta_t, viol, y_pred, denom, intercept_new, coef_new_j,
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
        i = random_state.randint(n_samples-1)+1
        i = indices_samples[i]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)
        for j in range(n_components):
            mean[j] = z[j]

    for ii in range(n_samples):
        i = indices_samples[ii]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)

        # if normalize
        if mean is not None:
            normalize(z, mean, var, t[0], n_components)

        y_pred = 0
        norm = 0
        for j in range(n_components):
            y_pred += z[j] * coef[j]
            acc_loss[0] += 0.5*lam2*coef[j]**2 + lam1*fabs(coef[j])

        y_pred += intercept[0]
        acc_loss[0] += loss.loss(y_pred, y[i])

        # update parameters
        dloss = loss.dloss(y_pred, y[i])

        eta_t = eta*t[0]
        if dloss != 0:
            for j in range(n_components):
                acc_grad[j] += dloss * z[j]
                acc_grad_norm[j] += (dloss*z[j])**2

        for j in range(n_components):
            denom = sqrt(acc_grad_norm[j]) + eps + lam2*eta_t
            if fabs(acc_grad[j])/t[0] - lam1 < 0:
                coef_new_j = 0
            else:
                coef_new_j = -eta_t / denom
                if acc_grad[j] > 0:
                    coef_new_j *= (acc_grad[j]/t[0] - lam1)
                else:
                    coef_new_j *= (acc_grad[j]/t[0] + lam1)

            viol += fabs(coef[j] - coef_new_j)
            coef[j] = coef_new_j

        if fit_intercept:
            acc_grad_intercept[0] += dloss
            acc_grad_norm_intercept[0] += dloss*dloss
            denom = sqrt(acc_grad_norm_intercept[0]) + eps

            intercept_new = -eta*acc_grad_intercept[0] / denom
            viol += fabs(intercept_new - intercept[0])
            intercept[0] = intercept_new

        t[0] += 1
    acc_loss[0] /= n_samples
    return viol


def _adagrad_fast(double[:] coef,
                  double[:] intercept,
                  RowDataset X,
                  X_array,
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
        viol = adagrad_epoch(coef, intercept, X, X_array, y, acc_grad,
                             acc_grad_norm, acc_grad_intercept,
                             acc_grad_norm_intercept, mean, var, loss, lam1,
                             lam2, eta, &t, eps, is_sparse,
                             fit_intercept, shuffle, random_state, &acc_loss,
                             transformer, transformer_fast, indices_samples, z,
                             )
        if verbose:
            print("Iteration {} Violation {} Loss {}".format(it+1, viol,
                                                             acc_loss))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it+1))
            break

    return it