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
                           int t,
                           Py_ssize_t n_components,
                           double eps):
    cdef double mean_new
    cdef Py_ssize_t j
    for j in range(n_components):
        mean_new = mean[j] + (z[j] - mean[j]) / (t+1)
        var[j] = var[j] * (1-1./t)
        var[j] += (z[j] - mean[j])*(z[j] - mean_new) / t
        mean[j] = mean_new
        z[j] = (z[j] - mean[j]) / (eps + sqrt(var[j]))


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
                  double eps_normalize,
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
    cdef double dloss, eta_t, viol, y_pred, denom
    cdef double intercept_new, coef_new_j, lam1, lam2
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz
    lam1 = alpha * l1_ratio
    lam2 = alpha * (1-l1_ratio)
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

    if mean is not None and t == 1:
        i = random_state.randint(n_samples-1)+1
        i = indices_samples[i]
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transform(X, X_array, z, i, data, indices, n_nz, is_sparse, transformer,
                  id_transformer, random_weights, offset, orders, p_choice,
                  coefs_maclaurin, z_cache, hash_indices, hash_signs,
                  degree, kernel, anova)

        for j in range(n_components):
            mean[j] = z[j]

    for it in range(max_iter):
        viol = 0
        random_state.shuffle(indices_samples)

        for i in indices_samples:
            X.get_row_ptr(i, &indices, &data, &n_nz)
            transform(X, X_array, z, i, data, indices, n_nz, is_sparse,
                      transformer, id_transformer, random_weights, offset,
                      orders, p_choice, coefs_maclaurin, z_cache, hash_indices,
                      hash_signs, degree, kernel, anova)

            # if normalize
            if mean is not None:
                normalize(z, mean, var, t, n_components, eps)

            y_pred = 0
            norm = 0
            for j in range(n_components):
                y_pred += z[j] * coef[j]

            y_pred += intercept[0]

            # update parameters
            dloss = loss.dloss(y_pred, y[i])
            eta_t = eta*t
            if dloss != 0:
                for j in range(n_components):
                    acc_grad[j] += dloss * z[j]
                    acc_grad_norm[j] += (dloss*z[j])**2

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