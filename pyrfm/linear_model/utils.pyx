# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport cos, sin, sqrt
import numpy as np
cimport numpy as np
from ..random_feature.random_features_fast cimport BaseCRandomFeature


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
                    BaseCRandomFeature transformer_fast
                    ):
    cdef Py_ssize_t j, n_components
    if transformer_fast is None:
        if is_sparse:
            _z = transformer.transform(X_array[i])[0]
        else:
            _z = transformer.transform(np.atleast_2d(X_array[i]))[0]
        n_components = z.shape[0]
        for j in range(n_components):
            z[j] = _z[j]
    else:
        transformer_fast.transform(z, data, indices, n_nz)