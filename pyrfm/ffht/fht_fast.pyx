# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef extern from "fht.h":
    int fht_double(double *buf, int log_n)


cdef void  _fwht1d_fast(double* out, int degree, bint normalize):
    """
    cdef Py_ssize_t jj, j1, i2, k, step, m, n_block, offset
    cdef double tmp, norm
    n_block = 1
    m = 2**degree
    step = m
    for deg in range(degree, 0, -1):
        step = step >> 1
        offset = 0
        # n_block * step = 2 ** (degree-m) * d/(2**(degree-m)) = d
        for k in range(n_block):
            for jj in range(step):
                j1 = offset + jj
                j2 = j1 + step
                tmp = out[j1] + out[j2]
                out[j2] = out[j1] - out[j2]
                out[j1] = tmp
            offset += 2*step

        n_block = n_block << 1
    """
    fht_double(out, degree)
    if normalize:
        norm = sqrt(2**degree)
        for jj in range(2**degree):
            out[jj] /= norm


cdef void _fwht2d_fast(double[:, ::1] out, int degree, bint normalize):
    cdef Py_ssize_t n, i
    n = out.shape[0]
    for i in range(n):
        _fwht1d_fast(&out[i, 0], degree, normalize)


def fwht1d_fast(np.ndarray[double, ndim=1] x,
                bint normalize,
                bint inplace=False):
    cdef Py_ssize_t n_features, n_features_padded
    cdef int degree
    cdef double* out_ptr
    cdef np.ndarray[double, ndim=1] out
    n_features = x.shape[0]
    degree = (n_features-1).bit_length()
    n_features_padded = 2 ** degree
    if not inplace:
        out = np.zeros(n_features_padded)
        out[:n_features] += x
    else:
        out = x
        if x.shape[0] != n_features_padded:
            raise ValueError("If inplace=True, len(x) must be power of 2.")
    out_ptr = <double *> out.data
    _fwht1d_fast(out_ptr, degree, normalize)
    return out


def fwht2d_fast(np.ndarray[double, ndim=2] X,
                bint normalize,
                bint inplace=False):
    cdef Py_ssize_t n, n_features, n_features_padded
    cdef int degree
    cdef double* out_ptr
    cdef np.ndarray[double, ndim=2] out

    n = X.shape[0]
    n_features = X.shape[1]
    degree = (n_features-1).bit_length()
    n_features_padded = 2 ** degree
    if not inplace:
        out = np.zeros((n, n_features_padded))
        out[:, :n_features] += X
    else:
        out = X
        if X.shape[1] != n_features_padded:
            raise ValueError("If inplace = True, "
                             "X.shape[1] must be power of 2.")
    _fwht2d_fast(out, degree, normalize)
    return out


def fwht(x, bint normalize=True, bint inplace=False):
    type = x.dtype
    if x.ndim == 1:
        ret = fwht1d_fast(x, normalize, inplace)
    elif x.ndim == 2:
        ret = fwht2d_fast(x, normalize, inplace)
    return ret