# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from libc.math cimport fabs, sqrt, pow
from .loss_fast cimport LossFunction
import numpy as np
cimport numpy as np
from ..dataset_fast cimport RowDataset
from .utils cimport transform, normalize
from cython.view cimport array
from ..random_feature.random_features_fast cimport BaseCRandomFeature


cdef inline double proximal(double coef,
                            double lam):
    if coef > lam:
        return coef - lam
    elif coef < -lam:
        return coef + lam
    else:
        return 0.


cdef inline double _get_eta(double eta0,
                            int learning_rate,
                            double power_t,
                            double lam2,
                            int t):
    cdef double eta = eta0
    if learning_rate == 1:
        eta = 1.0 / (lam2 * t)
    elif learning_rate == 2:
        eta /= pow(t, power_t)
    elif learning_rate == 3:
        eta /= pow(1+lam2*eta0*t, power_t)
    return eta


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


cdef double sgd_initialization(double[:] coef,
                               double[:] intercept,
                               double[:] averaged_grad_coef,
                               double[:] averaged_grad_intercept,
                               double[:] dloss_prev,
                               RowDataset X,
                               X_array,
                               double[:] y,
                               double[:] mean,
                               double[:] var,
                               LossFunction loss,
                               double lam1,
                               double lam2,
                               double intercept_decay,
                               double eta0,
                               int learning_rate,
                               double power_t,
                               unsigned int* t,
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
    cdef double dloss, eta_t, viol, y_pred, intercept_new, coef_new_j, update
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz

    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    viol = 0
    if mean is not None:
        for i in range(n_samples):
            X.get_row_ptr(i, &indices, &data, &n_nz)
            transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                      transformer_fast)
            for j in range(n_components):
                mean[j] += z[j] / n_samples
        for i in range(n_samples):
            X.get_row_ptr(i, &indices, &data, &n_nz)
            transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                      transformer_fast)
            for j in range(n_components):
                var[j] += (z[j] - mean[j])**2 / (n_samples-1)

    if shuffle:
        random_state.shuffle(indices_samples)
    for ii in range(n_samples):
        i = indices_samples[ii]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)

        # if normalize
        if mean is not None:
            for j in range(n_components):
                z[j] = (z[j] - mean[j]) / (sqrt(var[j])+1e-6)

        y_pred = _pred(z, coef, intercept[0], lam1, lam2,
                       &acc_loss[0], n_components)
        acc_loss[0] += loss.loss(y_pred, y[i])

        # update parameters
        dloss = loss.dloss(y_pred, y[i])
        dloss_prev[i] = dloss
        eta_t = _get_eta(eta0, learning_rate, power_t, lam2, t[0])
        for j in range(n_components):
            update = eta_t * (dloss*z[j] + lam2*coef[j])
            averaged_grad_coef[j] += dloss*z[j] / n_samples
            coef_new_j = coef[j] - update
            coef_new_j = proximal(coef_new_j, lam1*eta_t)
            viol += fabs(coef[j] - coef_new_j)
            coef[j] = coef_new_j

        if fit_intercept:
            eta_t = _get_eta(eta0, learning_rate, power_t,
                             intercept_decay, t[0])
            update = eta_t * (dloss + intercept_decay*intercept[0])
            averaged_grad_intercept[0] += dloss / n_samples
            intercept[0] -= update
            viol += fabs(update)

        t[0] += 1
    acc_loss[0] /= n_samples
    return viol


cdef double saga_epoch(double[:] coef,
                       double[:] intercept,
                       double[:] averaged_grad_coef,
                       double[:] averaged_grad_intercept,
                       double[:] dloss_prev,
                       RowDataset X,
                       X_array,
                       double[:] y,
                       double[:] mean,
                       double[:] var,
                       LossFunction loss,
                       double lam1,
                       double lam2,
                       double intercept_decay,
                       double eta0,
                       int learning_rate,
                       double power_t,
                       unsigned int denom,
                       unsigned int* t,
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
    cdef double dloss, eta_t, viol, y_pred, intercept_new, coef_new_j
    cdef double dloss_dif, update
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz

    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    viol = 0
    if shuffle:
        random_state.shuffle(indices_samples)
    for ii in range(n_samples):
        i = indices_samples[ii]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)
        # if normalize
        if mean is not None:
            for j in range(n_components):
                z[j] = (z[j] - mean[j]) / (sqrt(var[j])+1e-6)
    
        y_pred = _pred(z, coef, intercept[0], lam1, lam2,
                       &acc_loss[0], n_components)
        acc_loss[0] += loss.loss(y_pred, y[i])

        # update parameters
        dloss = loss.dloss(y_pred, y[i])
        dloss_dif = dloss - dloss_prev[i]
        dloss_prev[i] = dloss
        eta_t = _get_eta(eta0, learning_rate, power_t, lam2, t[0])
        for j in range(n_components):
            update = dloss_dif * z[j] / denom + averaged_grad_coef[j]
            update = eta_t * (update + lam2*coef[j])
            averaged_grad_coef[j] += dloss_dif * z[j] / n_samples
            coef_new_j = coef[j] - update
            coef_new_j = proximal(coef_new_j, lam1*eta_t)
            viol += fabs(coef[j] - coef_new_j)
            coef[j] = coef_new_j

        if fit_intercept:
            eta_t = _get_eta(eta0, learning_rate, power_t,
                             intercept_decay, t[0])
            update = dloss_dif / denom + averaged_grad_intercept[0]
            update = eta_t * (update + intercept_decay*intercept[0])
            averaged_grad_intercept[0] += dloss_dif / n_samples
            intercept[0] -= update
            viol += fabs(update)

        t[0] += 1
    acc_loss[0] /= n_samples
    return viol


def _saga_fast(double[:] coef,
               double[:] intercept,
               double[:] averaged_grad_coef,
               double[:] averaged_grad_intercept,
               double[:] dloss_prev,
               RowDataset X,
               X_array,
               double[:] y,
               double[:] mean,
               double[:] var,
               LossFunction loss,
               double alpha,
               double l1_ratio,
               double intercept_decay,
               double eta0,
               int learning_rate,
               double power_t,
               bint is_saga,
               unsigned int t,
               unsigned int max_iter,
               double tol,
               bint is_sparse,
               bint verbose,
               bint fit_intercept,
               bint shuffle,
               random_state,
               transformer,
               BaseCRandomFeature transformer_fast):
    cdef Py_ssize_t it, n_samples, n_components, j
    cdef double viol, lam1, lam2, acc_loss
    cdef unsigned int denom
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    if is_saga:
        denom = 1
    else:
        denom = n_samples

    cdef np.ndarray[int, ndim=1] indices_samples = np.arange(n_samples,
                                                             dtype=np.int32)
    cdef double[:] z = array((n_components, ), sizeof(double), format='d')
    for j in range(n_components):
        z[j] = 0

    it = 0
    if t == 1:
        acc_loss = 0
        viol = sgd_initialization(coef, intercept, averaged_grad_coef,
                                  averaged_grad_intercept, dloss_prev, X,
                                  X_array, y, mean, var, loss, lam1, lam2,
                                  intercept_decay, eta0, learning_rate,
                                  power_t, &t, is_sparse, fit_intercept,
                                  shuffle, random_state, &acc_loss,
                                  transformer, transformer_fast,
                                  indices_samples, z)
        if verbose:
            print("SGD Initialization. " 
                  "Violation {} Loss {}".format(viol, acc_loss))
    for it in range(max_iter):
        acc_loss = 0
        viol = saga_epoch(coef, intercept, averaged_grad_coef,
                          averaged_grad_intercept, dloss_prev, X, X_array, y,
                          mean, var, loss, lam1, lam2, intercept_decay, eta0,
                          learning_rate, power_t, denom, &t, is_sparse,
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