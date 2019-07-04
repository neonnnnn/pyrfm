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
from .random_mapping cimport (random_fourier, random_maclaurin, tensor_sketch,
                              random_kernel)
from cython.view cimport array


cdef inline void normalize(double[:] z,
                           double[:] mean,
                           double[:] var,
                           unsigned int t,
                           Py_ssize_t n_components):
    cdef double mean_new
    cdef Py_ssize_t j
    for j in range(n_components):
        mean_new = mean[j] + (z[j] - mean[j]) / (t+1)
        var[j] = var[j] * (1-1./t)
        var[j] += (z[j] - mean[j])*(z[j] - mean_new) / t
        mean[j] = mean_new
        z[j] = (z[j] - mean[j]) / (1e-6 + sqrt(var[j]))


cdef inline void transform(RowDataset X,
                           X_array,
                           double[:] z,
                           Py_ssize_t i,
                           double* data,
                           int* indices,
                           int n_nz,
                           bint is_sparse,
                           transformer,
                           int id_transformer,
                           double[:, ::1] random_weights,
                           double[:] offset,
                           int[:] orders,
                           double[:] p_choice,
                           double[:] coefs_maclaurin,
                           double[:] z_cache,
                           int[:] hash_indices,
                           int[:] hash_signs,
                           int degree,
                           int kernel,
                           double[:] anova,
                           ):
    cdef Py_ssize_t j
    if id_transformer == -1:
        if is_sparse:
            _z = transformer.transform(X_array[i])[0]
        else:
            _z = transformer.transform(np.atleast_2d(X_array[i]))[0]
        for j in range(z.shape[0]):
            z[j] = _z[j]
    else:
        if id_transformer == 0:
            random_fourier(z, data, indices, n_nz, random_weights, offset)
        elif id_transformer == 1:
            random_maclaurin(z, data, indices, n_nz, random_weights,
                             orders, p_choice, coefs_maclaurin)
        elif id_transformer == 2:
            tensor_sketch(z, z_cache, data, indices, n_nz, degree,
                          hash_indices, hash_signs)
        elif id_transformer == 3:
            random_kernel(z, data, indices, n_nz, random_weights, kernel,
                          degree, anova)
        else:
            raise ValueError("Random feature mapping must be RandomFourier,"
                             "RandomMaclaurin, TensorSketch, or "
                             "RandomKernel.")


cdef inline double proximal(double coef,
                            double lam):
    if coef > lam:
        return coef - lam
    elif coef < lam:
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
                                int id_transformer,
                                int[:] indices_samples,
                                double[:] z,
                                double[:, ::1] random_weights,
                                double[:] offset,
                                int[:] orders,
                                double[:] p_choice,
                                double[:] coefs_maclaurin,
                                double[:] z_cache,
                                int[:] hash_indices,
                                int[:] hash_signs,
                                int degree,
                                int kernel,
                                double[:] anova,
                                random_state):
    cdef Py_ssize_t i, j, n_samples, n_components
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
        transform(X, X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  id_transformer, random_weights, offset, orders, p_choice,
                  coefs_maclaurin, z_cache, hash_indices, hash_signs,
                  degree, kernel, anova)

        for j in range(n_components):
            mean[j] = z[j]

    # run modified SGD
    for i in indices_samples:
        X.get_row_ptr(i, &indices, &data, &n_nz)
        # compute random feature
        transform(X, X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  id_transformer, random_weights, offset, orders, p_choice,
                  coefs_maclaurin, z_cache, hash_indices, hash_signs,
                  degree, kernel, anova)

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
        update = loss.sdca_update(dual_coef[i], y[i], y_pred, norm/(lam2*t[0]))
        dual_coef[i] += update

        # update primal coef
        y_pred = 0
        # update primal coef
        for j in range(n_components):
            coef_old = coef[j]
            coef[j] *= lam2*(t[0]-1)
            coef[j] += update * z[j]
            coef[j] /= lam2*t[0]
            # proximal
            coef[j] = proximal(coef[j], lam1)
            y_pred +=  coef[j]*z[j]
            gap += lam2*coef[j]**2 + 2*lam1*fabs(coef[j])

        if fit_intercept:
            intercept[0] *= lam2*(t[0]-1)
            intercept[0] += update
            intercept[0] /= lam2*t[0]

        # compute duality gap
        gap += loss.loss(y_pred, y[i]) + loss.conjugate(-dual_coef[i], y[i])
        t[0] += 1
    return fabs(gap/n_samples)


cdef inline double _sdca_epoch(double[:] coef,
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
                               int id_transformer,
                               int[:] indices_samples,
                               double[:] z,
                               double[:, ::1] random_weights,
                               double[:] offset,
                               int[:] orders,
                               double[:] p_choice,
                               double[:] coefs_maclaurin,
                               double[:] z_cache,
                               int[:] hash_indices,
                               int[:] hash_signs,
                               int degree,
                               int kernel,
                               double[:] anova
                               ):
    cdef Py_ssize_t i, j, n_samples, n_components
    cdef double y_pred, update, norm, gap
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz
    gap = 0
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    for i in indices_samples:
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X, X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  id_transformer, random_weights, offset, orders, p_choice,
                  coefs_maclaurin, z_cache, hash_indices, hash_signs,
                  degree, kernel, anova)

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
        update = loss.sdca_update(dual_coef[i], y[i], y_pred,
                                  norm / (lam2*n_samples))
        dual_coef[i] += update

        # update primal coef
        y_pred = 0
        for j in range(n_components):
            coef_old = coef[j]
            coef[j] += update * z[j] / (n_samples*lam2)
            # proximal
            coef[j] = proximal(coef[j], lam1)
            y_pred += coef[j]*z[j]
            gap += lam2*coef[j]**2 + 2*lam1*fabs(coef[j])
        if fit_intercept:
            intercept[0] += update / (n_samples*lam2)
            y_pred += intercept[0]

        # compute duality gap
        gap += loss.loss(y_pred, y[i]) + loss.conjugate(-dual_coef[i], y[i])
        t[0] += 1
    return fabs(gap/n_samples)


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
               random_state,
               transformer,
               int id_transformer,
               double[:, ::1] random_weights,
               double[:] offset,
               int[:] orders,
               double[:] p_choice,
               double[:] coefs_maclaurin,
               int[:] hash_indices,
               int[:] hash_signs,
               int degree,
               int kernel,
               ):
    cdef Py_ssize_t it, i, n_samples, n_components, j
    cdef double gap, lam1, lam2
    lam1 = alpha*l1_ratio
    lam2 = alpha*(1-l1_ratio)
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]

    cdef int[:] indices_samples = np.arange(n_samples, dtype=np.int32)
    cdef double[:] z = array((n_components, ), sizeof(double), format='d')
    for i in range(n_components):
        z[i] = 0
    cdef double[:] z_cache = None
    cdef double[:] anova = None
    if id_transformer == 2:
        z_cache = array((n_components, ), sizeof(double), format='d')
        for i in range(n_components):
            z_cache[i] = 0
    if id_transformer == 3 and kernel == 0:
        anova = array((degree+1, ), sizeof(double), format='d')
        for i in range(degree+1):
            anova[i] = 0
        anova[0] = 1

    it = 0

    # initialize by SGD if t == 1
    if t == 1:
        random_state.shuffle(indices_samples)
        gap = _sgd_initialization(coef, dual_coef, intercept, X, X_array, y,
                                  mean, var, loss, lam1, lam2, &t, tol,
                                  is_sparse, fit_intercept, transformer,
                                  id_transformer, indices_samples, z,
                                  random_weights, offset, orders, p_choice,
                                  coefs_maclaurin,
                                  z_cache, hash_indices, hash_signs, degree,
                                  kernel, anova, random_state)
        if verbose:
            print("SGD Initialization Done. Duality Gap {}".format(gap))

    # start epoch
    for it in range(max_iter):
        random_state.shuffle(indices_samples)
        gap = _sdca_epoch(coef, dual_coef, intercept, X, X_array, y, mean, var,
                          loss, lam1, lam2, &t, is_sparse, fit_intercept,
                          transformer, id_transformer, indices_samples, z,
                          random_weights, offset, orders, p_choice,
                          coefs_maclaurin, z_cache, hash_indices,
                          hash_signs, degree, kernel, anova)
        if verbose:
            print("Iteration {} Duality Gap {}".format(it+1, gap))
        if gap < tol:
            if verbose:
                print("Converged at iteration {}".format(it+1))
            break

    return it