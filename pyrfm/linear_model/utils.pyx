# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport cos, sin, sqrt
import numpy as np
cimport numpy as np
from ..random_feature.random_mapping cimport random_mapping


cdef void normalize(double[:] z,
                    double[:] mean,
                    double[:] var,
                    int t,
                    Py_ssize_t n_components):
    cdef double mean_new
    cdef Py_ssize_t j
    for j in range(n_components):
        mean_new = mean[j] + (z[j] - mean[j]) / (t+1)
        var[j] = var[j] * (1-1./t)
        var[j] += (z[j] - mean[j])*(z[j] - mean_new) / t
        mean[j] = mean_new
        z[j] = (z[j] - mean[j]) / (1e-6 + sqrt(var[j]))


cdef void transform(X_array,
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
        random_mapping(z, data, indices, n_nz, id_transformer, random_weights,
                       offset, orders, p_choice, coefs_maclaurin, z_cache,
                       hash_indices, hash_signs, degree, kernel, anova)