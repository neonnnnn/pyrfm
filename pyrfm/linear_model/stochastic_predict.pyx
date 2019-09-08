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
from ..random_feature.random_mapping cimport BaseCRandomFeature
from cython.view cimport array


def _predict_fast(double[:] coef,
                  RowDataset X,
                  double[:] y_pred,
                  double[:] mean,
                  double[:] var,
                  BaseCRandomFeature transformer_fast,
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

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(z, data, indices, n_nz)
        # if normalize
        if mean is not None:
            for j in range(n_components):
                z[j] = (z[j] - mean[j]) / (1e-6 + sqrt(var[j]))

        y_pred[i] = 0
        for j in range(n_components):
            y_pred[i] += z[j] * coef[j]
