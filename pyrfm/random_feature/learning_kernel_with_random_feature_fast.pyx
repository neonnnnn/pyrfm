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


def optimize_chi2(double[:] p,
                  double[:] v,
                  double lam,
                  int n_components):
    cdef Py_ssize_t i
    for i in range(n_components):
        p[i] = v[i]

    _proj_l1ball(p, 2*lam*n_components, n_components)
    for i in range(n_components):
        p[i] /= 2*lam*n_components


def optimize_kl(double[:] p,
                double[:] v,
                double lam,
                int n_components):
    cdef Py_ssize_t i
    cdef double v_max, scale
    v_max = v[0]
    for i in range(1, n_components):
        if v[i] > v_max:
            v_max = v[i]
    scale = 0
    for i in range(n_components):
        p[i] = exp((v[i] - v_max) / lam)
        scale += p[i]
    for i in range(n_components):
        p[i] /= scale


def optimize_tv(double[:] p,
                double[:] v,
                double lam,
                int n_components):
    raise NotImplementedError()


def proj_l1ball(v, z):
    projed = np.array(v)
    _proj_l1ball(projed, z, len(projed))
    return projed


def proj_l1ball_sort(v, z):
    projed = np.array(v)
    _proj_l1ball_sort(projed, z, len(projed))
    return projed
