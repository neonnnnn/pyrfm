# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs
from cython.view cimport array
from lightning.impl.dataset_fast cimport ColumnDataset, RowDataset
from .loss_fast cimport LossFunction


# (block) coordinate descent
# update W[1, j], \ldots, W[n_outputs, j] at the same time independently
# Hence, does not compute (n_outputs \times n_outputs) Hessian.
cdef double _cd_primal_epoch(double[:] coef,
                             ColumnDataset X,
                             double[:] y,
                             double[:] X_col_norms,
                             double[::] y_pred,
                             RowDataset H,
                             double alpha,
                             LossFunction loss,
                             int[:] index_ptr):
    cdef Py_ssize_t jj, j, ii, i, k
    cdef double sum_viol, coef_old, update, dloss
    cdef Py_ssize_t n_features
    # Data pointers
    cdef double *data
    cdef int *indices
    cdef int n_nz

    cdef double *data_H
    cdef int *indices_H
    cdef int n_nz_H

    n_features = coef.shape[0]
    sum_viol = 0

    for jj in range(n_features):
        j = index_ptr[jj]
        X.get_column_ptr(j, &indices, &data, &n_nz)
        H.get_row_ptr(j, &indices_H, &data_H, &n_nz_H)

        # inv_step size di
        inv_step_size = loss.mu * X_col_norms[j]
        update = 0

        for ii in range(n_nz):
            i = indices[ii]
            dloss = loss.dloss(y_pred[i], y[i])
            update += dloss * data[ii]
        for ii in range(n_nz_H):
            i = indices_H[ii]
            update += coef[i] * data_H[ii]

        inv_step_size += alpha*data_H[n_nz-2]

        # update w[j]
        coef_old = coef[j]
        update /= inv_step_size
        coef[j] -= update
        sum_viol += fabs(coef_old -  coef[j])

        # Synchronize
        for ii in range(n_nz):
            i = indices[ii]
            y_pred[i] += (coef[j] - coef_old)* data[ii]

    return sum_viol


def _cd_primal(double[:] coef,
               double intercept,
               ColumnDataset X,
               double[:] y,
               double[:] X_col_norms,
               double[:] y_pred,
               RowDataset H,
               double alpha,
               LossFunction loss,
               int max_iter,
               double tol,
               bint fit_intercept,
               rng,
               bint verbose):

    cdef Py_ssize_t it, i, n_features, n_samples
    cdef double viol, update
    cdef bint converged = False
    cdef int* index_ptr
    n_features = coef.shape[0]
    n_samples = X.get_n_samples()
    it = 0
    for it in range(max_iter):
        viol = 0
        viol = _cd_primal_epoch(coef, X, y, X_col_norms, y_pred, H, alpha,
                                loss, rng.permutation(n_features))
        if fit_intercept:
            update = 0
            for i in range(n_samples):
                update += loss.dloss(y_pred[i], y[i])
            update /= (n_samples * loss.mu)
            intercept -= update
            for i in range(n_samples):
                y_pred[i] -= update
            viol += fabs(update)

        if verbose:
            print("Iteration {} Violation {}".format(it, viol))

        if viol < tol:
            print('Converged at iteration {}'.format(it+1))
            converged = True


    return converged, it+1