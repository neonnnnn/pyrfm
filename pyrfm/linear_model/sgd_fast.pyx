# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from libc.math cimport fabs, sqrt, pow, fmax
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
    if learning_rate == 1: # pegasos
        eta = 1.0 / (lam2 * t)
    elif learning_rate == 2: # inv_scaling
        eta /= pow(t, power_t) # optimal
    elif learning_rate == 3:
        eta /= pow(1.0 + eta0*lam2*t, power_t)
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


cdef double sgd_epoch(double[:] coef,
                      double[:] intercept,
                      double[:] coef_cache,
                      double[:] intercept_cache,
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
                      bint average,
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
    cdef double dloss, eta_t, viol, y_pred, intercept_new, coef_new_j, mu_t
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

        y_pred = _pred(z, coef_cache, intercept_cache[0], lam1, lam2,
                       &acc_loss[0], n_components)
        acc_loss[0] += loss.loss(y_pred, y[i])

        # update parameters
        dloss = loss.dloss(y_pred, y[i])

        eta_t = _get_eta(eta0, learning_rate, power_t, lam2, t[0])
        for j in range(n_components):
            update = eta_t * (dloss*z[j] + lam2*coef_cache[j])
            coef_new_j = coef_cache[j] - update
            coef_new_j = proximal(coef_new_j, lam1*eta_t)
            viol += fabs(coef_cache[j] - coef_new_j)
            coef_cache[j] = coef_new_j

        if fit_intercept:
            eta_t = _get_eta(eta0, learning_rate, power_t,
                             intercept_decay, t[0])
            update = eta_t*(dloss + intercept_decay*intercept_cache[0])
            intercept_cache[0] -= update
            viol += fabs(update)

        if average:
            mu_t = 1.0 / fmax(1, fmax(t[0]-n_samples, t[0]-n_components))
            for j in range(n_components):
                coef[j] += (coef_cache[j] - coef[j]) * mu_t
            intercept[0] += (intercept_cache[0] - intercept[0]) * mu_t

        t[0] += 1
    acc_loss[0] /= n_samples
    return viol


def _sgd_fast(double[:] coef,
              double[:] intercept,
              double[:] coef_cache,
              double[:] intercept_cache,
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
              bint average,
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
        viol = sgd_epoch(coef, intercept, coef_cache, intercept_cache, X,
                         X_array, y, mean, var, loss, lam1, lam2,
                         intercept_decay, eta0, learning_rate, power_t, average,
                         &t, is_sparse, fit_intercept, shuffle, random_state,
                         &acc_loss, transformer, transformer_fast,
                         indices_samples, z,
                         )
        if verbose:
            print("Iteration {} Violation {} Loss {}".format(it+1, viol,
                                                             acc_loss))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it+1))
            break

    if not average:
        for j in range(n_components):
            coef[j] = coef_cache[j]
        intercept[0] = intercept_cache[0]

    return it