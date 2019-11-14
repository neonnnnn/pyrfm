# Author: Kyohei Atarashi
# License: BSD-2-Clause

# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from cython.view cimport array
from libc.stdlib cimport srand, rand
from libc.math cimport log, exp
from ..dataset_fast import get_dataset
from ..dataset_fast cimport RowDataset
from .random_features_fast import get_fast_random_feature
from .random_features_fast cimport BaseCRandomFeature
from sklearn.utils.extmath import safe_sparse_dot


cdef inline void _proj_l1ball_sort(np.ndarray[np.float64_t, ndim=1] v,
                                   double z, 
                                   int n):
    cdef Py_ssize_t j
    cdef int rho
    cdef double cumsum, theta
    cdef np.ndarray[np.float64_t, ndim=1] mu = -np.sort(-v)
    cumsum = 0
    rho = n
    if n == 1:
        v[0] = z
    else:
        for j in range(n):
            if mu[j] < (cumsum+mu[j] - z) / (j+1):
                rho = j
                break
            cumsum += mu[j]
        theta = (cumsum - z) / rho
    for j in range(n):
        v[j] = v[j] - theta
        if v[j] < 0:
            v[j] = 0
    

cdef inline void _proj_l1ball(double[:] v,
                              double z,
                              int n):
    cdef Py_ssize_t i, n_canditates, pivot_idx, j
    cdef int rho, n_uppers, n_lowers, offset
    cdef double cumsum, cumsum_cache, pivot, theta
    cdef np.ndarray[np.int32_t, ndim=1] candidates
    candidates = np.arange(2*n, dtype=np.int32)
    n_canditates = n
    cumsum = 0
    cumsum_cache = 0
    rho = 0
    n_uppers = 0
    offset = 0
    
    while n_canditates != 0:
        pivot_idx = candidates[offset+(rand() % n_canditates)]
        pivot = v[pivot_idx]
        n_uppers = 0
        n_lowers = 0
        cumsum_cache = 0
        for i in range(n_canditates):
            j = candidates[offset+i]
            if j != pivot_idx:
                if v[j] >= pivot:
                    cumsum_cache += v[j]
                    candidates[n_uppers] = j
                    n_uppers += 1
                else:
                    candidates[n+n_lowers] = j
                    n_lowers += 1
        # discard uppers from candidates
        if (cumsum + cumsum_cache) - (rho+n_uppers)*pivot < z:
            n_canditates = n_lowers
            offset = n
            cumsum += cumsum_cache + pivot
            candidates[n_uppers] = pivot_idx
            n_uppers += 1
            rho += n_uppers
        else: # discard lowers from candidates
            n_canditates = n_uppers
            offset = 0

    theta = (cumsum - z) / rho
    for i in range(n):
        v[i] = v[i] - theta
        if v[i] < 0:
            v[i] = 0


cdef _compute_X_trans_y(BaseCRandomFeature transformer_fast,
                        RowDataset dataset,
                        long[:] y):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_samples = dataset.get_n_samples()
    cdef int n_components = transformer_fast.n_components
    cdef double* data
    cdef int* indices
    cdef int n_nz
    cdef double[:] z = array((n_components, ), sizeof(double), format='d')
    cdef double[:] v = array((n_components, ), sizeof(double), format='d')
    for j in range(n_components):
        v[j] = 0
        z[j] = 0
    
    for i in range(n_samples):
        dataset.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(&z[0], data, indices, n_nz)
        for j in range(n_components):
            v[j] += y[i] * z[j]
            z[j] = 0
    
    for j in range(n_components):
        v[j] = n_components * (v[j])**2
    return np.asarray(v)


def proj_l1ball(v, z):
    projed = np.array(v)
    _proj_l1ball(projed, z, len(projed))
    return projed


def proj_l1ball_sort(v, z):
    projed = np.array(v)
    _proj_l1ball_sort(projed, z, len(projed))
    return projed


def compute_X_trans_y(transformer, X, y):
    transformer_fast = get_fast_random_feature(transformer)
    if transformer_fast is not None:
        dataset = get_dataset(X, 'c')
        return _compute_X_trans_y(transformer_fast, dataset, y)
    else:
        X_trans = transformer.transform(X)
        scale = np.sqrt(X_trans.shape[1])
        v = safe_sparse_dot(y, X_trans*scale, dense_output=True)**2
        return v
