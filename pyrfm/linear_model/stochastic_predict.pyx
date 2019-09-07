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
from ..random_feature.random_mapping cimport random_mapping
from cython.view cimport array


def _predict_fast(double[:] coef,
                  RowDataset X,
                  double[:] y_pred,
                  double[:] mean,
                  double[:] var,
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
                  int kernel
                  ):
    cdef Py_ssize_t n_samples, n_components, j
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]
    cdef double[:] z = array((n_components, ), sizeof(double), format='d')
    for j in range(n_components):
        z[j] = 0
    cdef double[:] z_cache = None
    cdef double[:] anova = None
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz

    if id_transformer == 2:
        z_cache = array((n_components, ), sizeof(double), format='d')
        for j in range(n_components):
            z_cache[j] = 0
    if id_transformer == 3 and kernel == 0:
        anova = array((degree+1, ), sizeof(double), format='d')
        for j in range(degree+1):
            anova[j] = 0
        anova[0] = 1

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        random_mapping(z, data, indices, n_nz,
                       id_transformer, random_weights, offset,
                       orders, p_choice, coefs_maclaurin, z_cache, hash_indices,
                       hash_signs, degree, kernel, anova)

        # if normalize
        if mean is not None:
            for j in range(n_components):
                z[j] = (z[j] - mean[j]) / (1e-6 + sqrt(var[j]))

        y_pred[i] = 0
        for j in range(n_components):
            y_pred[i] += z[j] * coef[j]
