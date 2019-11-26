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
from ..random_feature.random_features_fast import get_fast_random_feature
from ..random_feature.random_features_doubly cimport BaseCDoublyRandomFeature


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
                            double eta1,
                            double lam2,
                            int t):
    cdef double eta = eta0
    if learning_rate == 1: # pegasos
        eta = 1.0 / (lam2 * t)
    elif learning_rate == 2: # inv_scaling
        eta /= pow(t, power_t) 
    elif learning_rate == 3: # optimal
        eta /= pow(1.0 + eta0*lam2*t, power_t)
    elif learning_rate == 4: # original
        eta /= (1.0 + eta1*t)
    return eta


cdef inline double _update_old(double[:] coef,
                               double lam1,
                               double lam2,
                               double eta_t,
                               Py_ssize_t start,
                               Py_ssize_t stop):
    cdef Py_ssize_t j
    cdef double viol, coef_new_j
    viol = 0
    for j in range(start, stop):
        coef_new_j = (1-eta_t*lam2) * coef[j]
        coef_new_j = proximal(coef_new_j, lam1*eta_t)
        viol += fabs(coef[j] - coef_new_j)
        coef[j] = coef_new_j
    return viol

    
cdef inline double _update(double[:] grad,
                           double[:] coef,
                           double[:] intercept,
                           double lam1,
                           double lam2,
                           double intercept_decay,
                           double eta_t,
                           double dloss_mean,
                           bint fit_intercept,
                           Py_ssize_t n_components_old,
                           Py_ssize_t n_components_diff):
    cdef Py_ssize_t j
    cdef double update, viol
    viol = 0
    for j in range(n_components_diff):
        update = eta_t * grad[j]
        viol += fabs(update)
        coef[n_components_old+j] -= update

    if fit_intercept:
        update = eta_t * (dloss_mean+intercept_decay*intercept[0])
        intercept[0] -= update
        viol += fabs(update)
    return viol


cdef inline double _average_gradient(double[:] y,
                                     double[:, ::1] Z,
                                     double[:] y_pred,
                                     double[:] grad,
                                     int* indices_samples,
                                     int batch_size,
                                     int n_components_diff,
                                     double* acc_loss,
                                     LossFunction loss,
                                     ):
    cdef Py_ssize_t i, ii, j
    cdef double dloss, dloss_mean
    dloss_mean = 0
    # compute average gradients
    for ii in range(batch_size):
        i = indices_samples[ii]
        acc_loss[0] += loss.loss(y_pred[ii], y[i])
        dloss = loss.dloss(y_pred[ii], y[i])
        dloss_mean += dloss / batch_size
        for j in range(n_components_diff):
            grad[j] += dloss * Z[ii, j] / batch_size
    return dloss_mean


cdef double doubly_sgd_epoch(double[:] coef,
                             double[:] intercept,
                             RowDataset X,
                             double[:] y,
                             LossFunction loss,
                             double lam1,
                             double lam2,
                             double intercept_decay,
                             double eta0,
                             int learning_rate,
                             double power_t,
                             double eta1,
                             unsigned int* t,
                             unsigned int batch_size,
                             unsigned int n_bases_sampled,
                             bint fit_intercept,
                             bint shuffle,
                             random_state,
                             double* acc_loss,
                             int n_components_diff,
                             BaseCDoublyRandomFeature transformer_fast,
                             np.ndarray[int, ndim=1] indices_samples,
                             double[:, ::1] Z,
                             double[:] y_pred,
                             double[:] grad):
    cdef Py_ssize_t i, ii, j, it, offset
    cdef int n_samples, n_components, n_components_old 
    cdef int n_iter_per_epoch
    cdef double eta_t, viol, dloss_mean

    n_samples = X.get_n_samples()
    n_components_old = transformer_fast.get_n_components()
    n_iter_per_epoch = int((n_samples-1)/batch_size) + 1

    viol = 0
    if shuffle:
        random_state.shuffle(indices_samples)
    
    for it in range(n_iter_per_epoch):
        transformer_fast.inc_n_components(n_bases_sampled)
        n_components = transformer_fast.get_n_components()
        offset = it * batch_size
        if offset + batch_size > n_samples:
            offset = n_samples - batch_size
        transformer_fast.pred_batch(Z, y_pred, coef, intercept[0], X,
                                    &indices_samples[offset], batch_size,
                                    n_components_old, n_components)
    
        # compute average gradients
        dloss_mean = _average_gradient(y, Z, y_pred, grad, 
                                       &indices_samples[offset], batch_size,
                                       n_components_diff, &acc_loss[0],
                                       loss)
        # update parameters
        eta_t = _get_eta(eta0, learning_rate, power_t, eta1, lam2, t[0])
        viol += _update_old(coef, lam1, lam2, eta_t, 0, n_components_old)
        viol += _update(grad, coef, intercept, lam1, lam2, intercept_decay,
                        eta_t, dloss_mean, fit_intercept, n_components_old,
                        n_components_diff)
        # reset caches
        for j in range(n_components_diff):
            grad[j] = 0

        n_components_old = n_components
        t[0] += 1
    
    acc_loss[0] /= n_samples
    return viol


def _doubly_sgd_fast(double[:] coef,
                     double[:] intercept,
                     RowDataset X,
                     double[:] y,
                     LossFunction loss,
                     double alpha,
                     double l1_ratio,
                     double intercept_decay,
                     double eta0,
                     int learning_rate,
                     double power_t,
                     double eta1,
                     unsigned int t,
                     unsigned int max_iter,
                     unsigned int batch_size,
                     unsigned int n_bases_sampled,
                     double tol,
                     bint verbose,
                     bint fit_intercept,
                     bint shuffle,
                     random_state,
                     BaseCDoublyRandomFeature transformer_fast):
    cdef Py_ssize_t it, n_samples, n_components_diff, j
    cdef double viol, lam1, lam2, acc_loss
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
    n_samples = X.get_n_samples()

    transformer_fast.inc_n_components(n_bases_sampled)
    n_components_diff = transformer_fast.get_n_components()
    transformer_fast.dec_n_components(n_bases_sampled)
    n_components_diff -= transformer_fast.get_n_components()

    cdef np.ndarray[int, ndim=1] indices_samples = np.arange(n_samples,
                                                             dtype=np.int32)
    cdef double[:, ::1] Z = array((batch_size, n_components_diff),
                                   sizeof(double), format='d')
    cdef double[:] y_pred = array((batch_size,), sizeof(double), format='d')                               
    cdef double[:] grad = array((n_components_diff, ), sizeof(double), format='d')

    for j in range(n_components_diff):
        grad[j] = 0
    it = 0
    for it in range(max_iter):
        acc_loss = 0
        viol = doubly_sgd_epoch(coef, intercept, X, y, loss, lam1, lam2,
                                intercept_decay, eta0, learning_rate,
                                power_t, eta1, &t, batch_size, n_bases_sampled,
                                fit_intercept, shuffle, random_state,
                                &acc_loss, n_components_diff, transformer_fast,
                                indices_samples, Z, y_pred, grad)
        if verbose:
            print("Iteration {} Violation {} Loss {}".format(it+1, viol,
                                                             acc_loss))

        if viol < tol:
            if verbose:
                print("Converged at iteration {}".format(it+1))
            break
    
    return it
