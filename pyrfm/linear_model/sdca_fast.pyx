# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs, sqrt
from .loss_fast cimport LossFunction
import numpy as np
cimport numpy as np
from lightning.impl.dataset_fast cimport RowDataset
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


cdef double _sgd_initialization(double[:] coef,
                                double[:] dual_coef,
                                double[:] intercept,
                                RowDataset X,
                                X_array,
                                double[:] y,
                                double[:] mean,
                                double[:] var,
                                LossFunction loss,
                                double lam1,
                                double lam2,
                                unsigned int* t,
                                double tol,
                                bint is_sparse,
                                bint fit_intercept,
                                transformer,
                                BaseCRandomFeature transformer_fast,
                                np.ndarray[int, ndim=1] indices_samples,
                                double[:] z,
                                random_state):
    cdef Py_ssize_t i, ii, j
    cdef int n_samples, n_components
    cdef double y_pred, update, norm, gap
    cdef int* indices
    cdef double* data
    cdef int n_nz
    gap = 0.
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]
    if mean is not None and t[0] == 1:
        i = random_state.randint(n_samples-1)+1
        i = indices_samples[i]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)

        for j in range(n_components):
            mean[j] = z[j]

    # run modified SGD
    for ii in range(n_samples):
        i = indices_samples[ii]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        # compute random feature
        transform(X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  transformer_fast)
        # if normalize
        if mean is not None:
            normalize(z, mean, var, t[0], n_components)

        y_pred = 0
        norm = 0
        for j in range(n_components):
            y_pred += z[j] * coef[j]
            norm += z[j]**2
        if fit_intercept:
            y_pred += intercept[0]
            norm += 1

        # update dual_coef
        update = loss.sdca_update(dual_coef[i], y[i], y_pred, norm/(lam2*t[0]))
        dual_coef[i] += update
        # update primal coef
        y_pred = 0
        # update primal coef
        for j in range(n_components):
            coef[j] *= lam2*(t[0]-1)
            coef[j] += update * z[j]
            coef[j] /= lam2*t[0]
            # proximal
            coef[j] = proximal(coef[j], lam1)
            y_pred += coef[j]*z[j]

        if fit_intercept:
            intercept[0] *= lam2*(t[0]-1)
            intercept[0] += update
            intercept[0] /= lam2*t[0]

        # compute duality gap
        gap += loss.loss(y_pred, y[i]) + loss.conjugate(-dual_coef[i], y[i])
        t[0] += 1
    gap /= n_samples
    for j in range(n_components):
        gap += lam2*coef[j]**2 + 2*lam1*fabs(coef[j])

    return fabs(gap)


cdef double _sdca_epoch(double[:] coef,
                        double[:] dual_coef,
                        double[:] intercept,
                        RowDataset X,
                        X_array,
                        double[:] y,
                        double[:] mean,
                        double[:] var,
                        LossFunction loss,
                        double lam1,
                        double lam2,
                        unsigned int* t,
                        bint is_sparse,
                        bint fit_intercept,
                        transformer,
                        BaseCRandomFeature transformer_fast,
                        np.ndarray[int, ndim=1] indices_samples,
                        double[:] z,
                        ):
    cdef Py_ssize_t i, ii, j
    cdef int n_samples, n_components
    cdef double y_pred, update, norm, gap, scale
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz

    gap = 0
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]
    scale = 1./(lam2*n_samples)
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
            norm += z[j]**2

        y_pred += intercept[0]
        if fit_intercept:
            norm += 1

        # update dual_coef
        update = loss.sdca_update(dual_coef[i], y[i], y_pred, norm * scale)
        # update primal coef
        if update != 0:
            dual_coef[i] += update
            y_pred = 0
            for j in range(n_components):
                coef[j] += update * z[j] * scale
                # proximal
                coef[j] = proximal(coef[j], lam1)
                y_pred += coef[j] * z[j]

            if fit_intercept:
                intercept[0] += update * scale
                y_pred += intercept[0]

        # compute duality gap
        gap += loss.loss(y_pred, y[i]) + loss.conjugate(-dual_coef[i], y[i])
        t[0] += 1

    gap /= n_samples
    for j in range(n_components):
        gap += lam2 * coef[j]**2 + 2*lam1*fabs(coef[j])

    return fabs(gap)


def _sdca_fast(double[:] coef,
               double[:] dual_coef,
               double[:] intercept,
               RowDataset X,
               X_array,
               double[:] y,
               double[:] mean,
               double[:] var,
               LossFunction loss,
               double alpha,
               double l1_ratio,
               unsigned int t,
               unsigned int max_iter,
               double tol,
               bint is_sparse,
               bint verbose,
               bint fit_intercept,
               bint shuffle,
               random_state,
               transformer,
               BaseCRandomFeature transformer_fast
               ):
    cdef Py_ssize_t it, i, j
    cdef int n_samples, n_components
    cdef double gap, lam1, lam2
    lam1 = alpha*l1_ratio
    lam2 = alpha*(1-l1_ratio)
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    cdef np.ndarray[int, ndim=1] indices_samples = np.arange(n_samples,
                                                             dtype=np.int32)
    cdef double[:] z = array((n_components, ), sizeof(double), format='d')
    for i in range(n_components):
        z[i] = 0


    it = 0

    # initialize by SGD if t == 1
    if t == 1:
        if shuffle:
            random_state.shuffle(indices_samples)
        gap = _sgd_initialization(coef, dual_coef, intercept, X, X_array, y,
                                  mean, var, loss, lam1, lam2, &t, tol,
                                  is_sparse, fit_intercept, transformer,
                                  transformer_fast, indices_samples, z,
                                  random_state)
        if verbose:
            print("SGD Initialization Done. Duality Gap {}".format(gap))

    # start epoch
    for it in range(max_iter):
        if shuffle:
            random_state.shuffle(indices_samples)
        gap = _sdca_epoch(coef, dual_coef, intercept, X, X_array, y, mean, var,
                          loss, lam1, lam2, &t, is_sparse, fit_intercept,
                          transformer, transformer_fast, indices_samples, z)
        if verbose:
            print("Iteration {} Duality Gap {}".format(it+1, gap))
        if gap < tol:
            if verbose:
                print("Converged at iteration {}".format(it+1))
            break

    return it